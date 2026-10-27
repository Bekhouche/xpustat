"""
ui.py — a browser page for xpustat, served by the exporter's HTTP server.

``xpustat --ui`` starts the same server as ``--serve-prometheus`` and answers
``/`` with one self-contained HTML page (inline CSS and JavaScript, nothing
fetched from the network) that polls ``/cluster`` and draws every device it
knows about: memory, utilization, temperature, power, fan, clocks, ECC,
P-state, throttle reasons, health, the processes on each device, MIG
instances, the CPU row, and per-device sparklines for the open tab.

``/cluster`` returns ``{"interval": s, "hosts": [host, ...]}`` where each host
is a :class:`xpustat.remote.RemoteHost` dict plus ``stale`` and ``updated``.
Without remote hosts it holds the local machine only. With ``--hosts``,
``--ssh`` or ``--hosts-file`` a :class:`ClusterCollector` refreshes the
remote payloads on its own timer, so every open tab reads one cached
fan-out instead of starting its own. A host that stops answering keeps its
last good snapshot and is marked ``stale`` with the error.

``/json`` and ``/metrics`` stay available on the same port, for scripts and
Prometheus.
"""

from __future__ import annotations

import json
import socket
import sys
import threading
import time
from http.server import ThreadingHTTPServer
from typing import Any, Callable, Dict, List, Optional

from .prometheus import _Handler
from .remote import RemoteHost, query_hosts, query_ssh


# ---------------------------------------------------------------------------
# Host lists
# ---------------------------------------------------------------------------

def read_hosts_file(path: str) -> List[str]:
    """One host per line; blank lines and ``#`` comments ignored."""
    hosts: List[str] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.split("#", 1)[0].strip()
            if line:
                hosts.append(line)
    return hosts


def split_hosts(entries: List[str]) -> "tuple[List[str], List[str]]":
    """Exporter URLs (``http://``, ``https://`` or ``host:port``) versus ssh hosts."""
    urls, ssh = [], []
    for e in entries:
        if e.startswith(("http://", "https://")):
            urls.append(e)
        elif ":" in e and e.rsplit(":", 1)[1].isdigit() and "@" not in e:
            urls.append(e)
        else:
            ssh.append(e)
    return urls, ssh


# ---------------------------------------------------------------------------
# Cluster collector
# ---------------------------------------------------------------------------

class ClusterCollector:
    """
    Fetch the remote payloads on a timer and keep the latest per host.

    ``fetch`` may be given for tests; it returns a list of :class:`RemoteHost`.
    """

    def __init__(self, urls: Optional[List[str]] = None, ssh: Optional[List[str]] = None,
                 command: Optional[str] = None, include_cpu: bool = False,
                 interval: float = 5.0, timeout: float = 15.0,
                 fetch: Optional[Callable[[], List[RemoteHost]]] = None,
                 local: Optional[Callable[[], Any]] = None) -> None:
        self.urls = list(urls or [])
        self.ssh = list(ssh or [])
        self.command = command
        self.include_cpu = include_cpu
        self.interval = max(0.5, float(interval))
        self.timeout = timeout
        self._fetch = fetch
        self.local = local  # callable returning an XPUStatCollection for this machine, or None
        self._hosts: Dict[str, dict] = {}
        self._order: List[str] = []
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.updated: float = 0.0
        self.refreshes = 0

    @property
    def labels(self) -> List[str]:
        return (["local"] if self.local else []) + self.urls + self.ssh

    def _local_host(self) -> RemoteHost:
        try:
            h = local_payload(self.local(), self.interval)["hosts"][0]
        except Exception as exc:
            return RemoteHost("local", error=f"{type(exc).__name__}: {exc}")
        return RemoteHost("local", devices=h["devices"], hostname=h["hostname"],
                          query_time=h["query_time"], version=h["xpustat_version"])

    def fetch(self) -> List[RemoteHost]:
        out: List[RemoteHost] = [self._local_host()] if self.local else []
        if self._fetch is not None:
            return out + self._fetch()
        if self.urls:
            out += query_hosts(self.urls, timeout=self.timeout)
        if self.ssh:
            out += query_ssh(self.ssh, command=self.command, include_cpu=self.include_cpu, timeout=self.timeout)
        return out

    def refresh(self) -> None:
        """One synchronous fan-out; dead hosts keep their last good snapshot."""
        try:
            results = self.fetch()
        except Exception as exc:  # the collector thread must survive anything
            results = [RemoteHost(label, error=f"{type(exc).__name__}: {exc}") for label in self.labels]
        now = time.time()
        with self._lock:
            for h in results:
                if h.label not in self._order:
                    self._order.append(h.label)
                prev = self._hosts.get(h.label)
                if h.ok:
                    entry = h.to_dict()
                    entry.update(stale=False, updated=now)
                elif prev and prev.get("devices"):
                    entry = dict(prev)
                    entry.update(error=h.error, stale=True)
                else:
                    entry = h.to_dict()
                    entry.update(stale=False, updated=now)
                self._hosts[h.label] = entry
            self.updated = now
            self.refreshes += 1

    def snapshot(self) -> dict:
        with self._lock:
            hosts = [self._hosts[k] for k in self._order if k in self._hosts]
            return {"interval": self.interval, "updated": self.updated, "hosts": hosts,
                    "cluster": True, "refreshes": self.refreshes, "version": _version()}

    def _loop(self) -> None:
        while not self._stop.is_set():
            self.refresh()
            self._stop.wait(self.interval)

    def start(self) -> "ClusterCollector":
        if self._thread is None:
            self._thread = threading.Thread(target=self._loop, name="xpustat-cluster", daemon=True)
            self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)
            self._thread = None


# ---------------------------------------------------------------------------
# HTTP handler
# ---------------------------------------------------------------------------

def local_payload(stats: Any, interval: float) -> dict:
    """``/cluster`` shape for the local machine only."""
    d = json.loads(stats.to_json(metadata=True))
    host = {"host": d.get("hostname") or socket.gethostname(), "hostname": d.get("hostname"),
            "query_time": d.get("query_time"), "xpustat_version": d.get("xpustat_version"),
            "error": None, "devices": d.get("devices", {}), "stale": False, "updated": time.time()}
    return {"interval": interval, "updated": host["updated"], "hosts": [host], "cluster": False,
            "refreshes": 0, "version": _version()}


def _version() -> str:
    from . import __version__
    return __version__


class _UIHandler(_Handler):
    cluster: Optional[ClusterCollector] = None
    interval: float = 2.0
    page: bytes = b""

    def do_GET(self) -> None:  # noqa: N802
        path = self.path.split("?", 1)[0]
        if path in ("/", "/index.html"):
            self._send(200, "text/html; charset=utf-8", type(self).page)
            return
        if path == "/cluster":
            try:
                col = type(self).cluster
                if col is not None:
                    payload = col.snapshot()
                else:
                    payload = local_payload(type(self).query(), type(self).interval)
                body = json.dumps(payload).encode("utf-8")
                status = 200
            except Exception as exc:
                body = json.dumps({"error": f"{type(exc).__name__}: {exc}"}).encode("utf-8")
                status = 500
            self._send(status, "application/json; charset=utf-8", body)
            return
        super().do_GET()

    def _send(self, status: int, ctype: str, body: bytes) -> None:
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)


def make_handler(query: Callable[[], Any], cluster: Optional[ClusterCollector] = None,
                 interval: float = 2.0) -> type:
    return type("XPUStatUIHandler", (_UIHandler,),
                {"query": staticmethod(query), "cluster": cluster, "interval": float(interval),
                 "page": PAGE.encode("utf-8")})


def serve_ui(port: int = 9400, host: str = "127.0.0.1", query: Optional[Callable[[], Any]] = None,
             max_age: float = 0.0, include_cpu: bool = False,
             cluster: Optional[ClusterCollector] = None, interval: float = 2.0) -> None:
    """
    Serve the page at ``/``, plus ``/cluster``, ``/json`` and ``/metrics``, forever.

    Args:
        port, host: bind address; the UI defaults to loopback because the page
            shows process command lines.
        query: callable returning an ``XPUStatCollection`` (default ``query_all``).
        max_age: reuse a local result up to this many seconds old.
        include_cpu: include the host CPU in the local query.
        cluster: a started :class:`ClusterCollector` for the multi-host page.
        interval: the page's poll interval in seconds (local mode).
    """
    if query is None:
        from .xpu import query_all
        query = lambda: query_all(max_age=max_age if max_age > 0 else None, include_cpu=include_cpu)
    handler = make_handler(query, cluster, cluster.interval if cluster else interval)
    server = ThreadingHTTPServer((host, port), handler)
    shown = "localhost" if host in ("0.0.0.0", "::", "") else host
    sys.stderr.write(f"xpustat: web UI on http://{shown}:{port}/  (/json, /metrics and /cluster on the same port)\n")
    if cluster is not None:
        if cluster.local is True:  # the CLI asks for the local machine; wire the query in
            cluster.local = query
        sys.stderr.write(f"xpustat: polling {len(cluster.labels)} host(s) every {cluster.interval:g} s\n")
        cluster.start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        if cluster is not None:
            cluster.stop()
        server.server_close()


# ---------------------------------------------------------------------------
# The page
# ---------------------------------------------------------------------------

PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>xpustat</title>
<style>
:root{--bg:#f6f7f9;--panel:#fff;--line:#e3e6ea;--fg:#1d2126;--dim:#6b7380;--bar:#e9ecf0;--ok:#2e9e5b;--warn:#d9821f;--bad:#d03a3a;--mono:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;--sans:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
--nvidia:#5aa300;--amd:#d8262b;--gaudi:#1b9aaa;--intel:#1e6fd9;--huawei:#b3368e;--hygon:#c28a00;--cambricon:#0e8f9c;--moorethreads:#9b3fb5;--apple:#555;--neuron:#2f6ed1;--tpu:#2a9d4a;--tenstorrent:#b58a00;--jetson:#5aa300;--cpu:#7a8290;--rebellions:#d45049;--furiosa:#0e8f9c;--other:#7a8290}
@media (prefers-color-scheme:dark){:root{--bg:#121417;--panel:#1a1d22;--line:#2a2f37;--fg:#e6e8eb;--dim:#8d95a1;--bar:#262b33;--apple:#b9bec6}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.45 var(--sans)}
header{display:flex;flex-wrap:wrap;gap:10px 18px;align-items:center;padding:10px 16px;border-bottom:1px solid var(--line);background:var(--panel);position:sticky;top:0;z-index:2}
header h1{font-size:16px;margin:0;font-weight:650}header h1 span{color:var(--dim);font-weight:400;margin-left:6px}
header .ctl{display:flex;gap:8px;align-items:center;flex-wrap:wrap;margin-left:auto;font-size:13px}
select,input[type=search],button{font:inherit;color:inherit;background:var(--bg);border:1px solid var(--line);border-radius:6px;padding:4px 8px}
button{cursor:pointer}button.on{border-color:var(--fg)}
label{display:inline-flex;align-items:center;gap:4px;cursor:pointer}
#status{color:var(--dim);font-size:12px;font-family:var(--mono)}
.summary{display:grid;grid-template-columns:repeat(auto-fit,minmax(130px,1fr));gap:8px;padding:12px 16px}
.tile{background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:8px 10px}
.tile .k{font-size:11px;color:var(--dim);text-transform:uppercase;letter-spacing:.04em}.tile .v{font-size:20px;font-weight:650;font-family:var(--mono)}.tile .s{font-size:12px;color:var(--dim);white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
main{padding:0 16px 24px}
.host{background:var(--panel);border:1px solid var(--line);border-radius:8px;margin:0 0 12px}
.host>.hh{display:flex;gap:12px;align-items:baseline;padding:8px 12px;border-bottom:1px solid var(--line);font-family:var(--mono);font-size:13px}
.host>.hh b{font-size:14px}.host>.hh .dim{color:var(--dim)}.host.dead>.hh b{color:var(--bad)}.host.stale>.hh b{color:var(--warn)}
.err{color:var(--bad);padding:8px 12px;font-family:var(--mono);font-size:13px}.none{color:var(--dim);padding:8px 12px}
table{width:100%;border-collapse:collapse;font-family:var(--mono);font-size:13px}
th{font-weight:500;color:var(--dim);text-align:left;padding:5px 8px;font-size:11px;text-transform:uppercase;letter-spacing:.03em;white-space:nowrap}
td{padding:4px 8px;border-top:1px solid var(--line);white-space:nowrap;vertical-align:middle}
tr.dev{cursor:pointer}tr.dev:hover td{background:color-mix(in srgb,var(--fg) 4%,transparent)}
tr.mig td{color:var(--dim)}tr.mig td:first-child{padding-left:28px}
.tag{display:inline-block;min-width:58px;padding:1px 6px;border-radius:4px;color:#fff;font-size:11px;font-weight:650;text-align:center;letter-spacing:.02em}
.idx{font-weight:650;margin-left:6px}
.name{max-width:260px;overflow:hidden;text-overflow:ellipsis}
.bar{display:inline-block;width:120px;height:9px;background:var(--bar);border-radius:3px;vertical-align:middle;overflow:hidden;margin-right:6px}.bar i{display:block;height:100%;border-radius:3px}
.num{text-align:right}.dim{color:var(--dim)}
.b{display:inline-block;padding:0 6px;border-radius:4px;font-size:11px;font-weight:650;line-height:18px}
.b.ok{background:color-mix(in srgb,var(--ok) 18%,transparent);color:var(--ok)}.b.warn{background:color-mix(in srgb,var(--warn) 18%,transparent);color:var(--warn)}.b.bad{background:color-mix(in srgb,var(--bad) 18%,transparent);color:var(--bad)}.b.na{background:var(--bar);color:var(--dim)}
.hot{color:var(--bad);font-weight:650}.warm{color:var(--warn)}
svg.sp{width:90px;height:22px;vertical-align:middle}
tr.det td{background:color-mix(in srgb,var(--fg) 3%,transparent);white-space:normal;padding:6px 8px 8px 40px}
.procs{width:auto;font-size:12px}.procs th,.procs td{padding:2px 8px;border:0}.procs td.cmd{max-width:520px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.kv{display:flex;flex-wrap:wrap;gap:4px 16px;font-size:12px;color:var(--dim)}.kv b{color:var(--fg);font-weight:500}
.compact .cx{display:none}
@media (max-width:900px){.cx{display:none}.name{max-width:140px}.bar{width:70px}}
</style>
</head>
<body>
<header>
  <h1>xpustat<span id="ver"></span></h1>
  <div id="status">connecting…</div>
  <div class="ctl">
    <select id="fHost" title="Host filter"><option value="">all hosts</option></select>
    <select id="fVendor" title="Vendor filter"><option value="">all vendors</option></select>
    <input id="fText" type="search" placeholder="filter: name, user, command, container" size="30">
    <label><input id="compact" type="checkbox"> compact</label>
    <button id="expandAll" title="Expand or collapse every device">expand all</button>
    <button id="pause">pause</button>
  </div>
</header>
<div class="summary" id="summary"></div>
<main id="main"></main>
<script>
(function(){
"use strict";
var VENDORS={nvidia:"NVIDIA",amd:"AMD",gaudi:"GAUDI",intel:"INTEL",huawei:"HUAWEI",hygon:"HYGON",cambricon:"CAMBR.",moorethreads:"MTT",apple:"APPLE",neuron:"NEURON",tpu:"TPU",tenstorrent:"TENST.",jetson:"JETSON",cpu:"CPU",rebellions:"RBLN",furiosa:"FURIOSA"};
var state={data:null,paused:false,expanded:{},hist:{},interval:2,timer:null,errors:0,expandNew:/[?&]expand\b/.test(location.search)};
var $=function(id){return document.getElementById(id)};
function esc(s){return String(s==null?"":s).replace(/[&<>"']/g,function(c){return {"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]})}
function num(v,d){return (v==null||isNaN(v))?null:Number(v).toFixed(d==null?0:d)}
function fmtInt(v){return v==null?"–":Math.round(v).toLocaleString()}
function first(d){for(var i=1;i<arguments.length;i++){var v=d[arguments[i]];if(v!=null)return v}return null}
function idxOf(d){var k=["gpu_index","aip_index","device_id","card_index","gpu_id","neuron_device","tt_device","npu_index","cpu_index"];for(var i=0;i<k.length;i++)if(d[k[i]]!=null)return String(d[k[i]]);if(d.npu_id!=null)return d.npu_id+":"+(d.chip_id||0);if(d.card_id!=null)return d.card_id+":"+(d.chip_id||0);return "?"}
function vcolor(v){return "var(--"+(VENDORS[v]?v:"other")+")"}
function bar(used,total,color){var p=total>0?Math.min(100,used/total*100):0;return '<span class="bar"><i style="width:'+p.toFixed(1)+'%;background:'+color+'"></i></span>'}
function spark(arr,max,color){if(!arr||arr.length<2)return '<svg class="sp"></svg>';var w=90,h=22,n=arr.length,pts=[];for(var i=0;i<n;i++){var v=arr[i]==null?0:arr[i];pts.push((i*(w-2)/(n-1)+1).toFixed(1)+","+(h-1-(max>0?v/max:0)*(h-2)).toFixed(1))}return '<svg class="sp" viewBox="0 0 '+w+' '+h+'" preserveAspectRatio="none"><polyline fill="none" stroke="'+color+'" stroke-width="1.5" points="'+pts.join(" ")+'"/></svg>'}
function tempCls(t){return t==null?"":t>=85?"hot":t>=70?"warm":""}
function health(d){var out=[];if(d.healthy===false)out.push('<span class="b bad" title="'+esc((d.throttle_reasons||[]).join(", "))+'">fault</span>');else if(d.healthy===true)out.push('<span class="b ok">ok</span>');else if(d.health!=null)out.push('<span class="b '+(String(d.health).toUpperCase()==="OK"?"ok":"warn")+'">'+esc(d.health)+'</span>');
if(d.throttled)out.push('<span class="b warn" title="'+esc((d.throttle_reasons||[]).join(", "))+'">throttled</span>');
if(d.ecc_uncorrected)out.push('<span class="b bad">ECC '+d.ecc_uncorrected+'</span>');
return out.join(" ")||'<span class="b na">–</span>'}
function isFree(d){var t=d.mem_total_mb||0,u=d.mem_used_mb||0,util=first(d,"util_pct","ai_core_pct");return t>0&&u<=t*0.05&&(util==null||util<5)&&!(d.processes&&d.processes.length)}
function pushHist(key,util,mem){var h=state.hist[key]||(state.hist[key]={u:[],m:[]});h.u.push(util==null?null:+util);h.m.push(mem==null?null:+mem);if(h.u.length>120){h.u.shift();h.m.shift()}return h}
function matches(d,text){if(!text)return true;text=text.toLowerCase();var hay=[d.name||"",idxOf(d)];(d.processes||[]).forEach(function(p){hay.push(p.username||"",p.command||"",p.cmdline||"",p.container||"",String(p.pid||""))});return hay.join(" ").toLowerCase().indexOf(text)>=0}

function render(){var data=state.data;if(!data)return;var hosts=data.hosts||[];
var fh=$("fHost").value,fv=$("fVendor").value,ft=$("fText").value.trim();
var vendorsSeen={},hostsSeen=[];var S={up:0,down:0,stale:0,dev:0,free:0,mu:0,mt:0,hot:null,thr:0,procs:0,fault:0};
var html="";
hosts.forEach(function(h){var label=h.hostname||h.host;hostsSeen.push(h.host);
if(h.error&&!h.stale){S.down++}else{S.up++;if(h.stale)S.stale++}
if(fh&&h.host!==fh)return;
var cls="host"+(h.error&&!h.stale?" dead":"")+(h.stale?" stale":"");
html+='<section class="'+cls+'"><div class="hh"><b>'+esc(label)+'</b>'+(h.host!==label?'<span class="dim">'+esc(h.host)+'</span>':"")+(h.xpustat_version?'<span class="dim">xpustat '+esc(h.xpustat_version)+'</span>':"")+(h.query_time?'<span class="dim">'+esc(h.query_time)+'</span>':"")+(h.stale?'<span class="b warn" title="'+esc(h.error)+'">stale: '+esc(h.error)+'</span>':"")+'</div>';
if(h.error&&!h.stale){html+='<div class="err">unreachable: '+esc(h.error)+'</div></section>';return}
var rows="";var devs=h.devices||{};
Object.keys(devs).forEach(function(v){var list=devs[v]||[];if(!list.length)return;vendorsSeen[v]=1;if(fv&&v!==fv)return;
list.forEach(function(d){var idx=idxOf(d),key=h.host+"|"+v+"|"+idx,color=vcolor(v);
var used=d.mem_used_mb||0,total=d.mem_total_mb||0,util=first(d,"util_pct","ai_core_pct");
if(!h.stale){S.dev++;if(v!=="cpu"){S.mu+=used;S.mt+=total}if(isFree(d))S.free++;if(d.throttled)S.thr++;if(d.healthy===false||d.ecc_uncorrected)S.fault++;S.procs+=(d.processes||[]).length;if(d.temp_c!=null&&(S.hot==null||d.temp_c>S.hot.t))S.hot={t:d.temp_c,l:label+" "+(VENDORS[v]||v)+" #"+idx}}
var hist=h.stale?(state.hist[key]||{u:[],m:[]}):pushHist(key,util,total>0?used/total*100:null);
if(!matches(d,ft))return;
if(state.expandNew&&!(key in state.expanded))state.expanded[key]=true;var open=!!state.expanded[key];
var clk=first(d,"clock_sm_mhz","clock_gfx_mhz","freq_mhz"),clkMax=first(d,"clock_max_sm_mhz","clock_max_gfx_mhz","freq_max_mhz"),cmem=d.clock_mem_mhz;
var fan=d.fan_pct!=null?d.fan_pct+"%":d.fan_rpm!=null?d.fan_rpm+" rpm":"–";
var arch=first(d,"cuda_cc","gfx_target","arch")||(d.gpu_cores!=null?d.gpu_cores+" cores":null)||(d.card_type||"");
var pst=first(d,"pstate","perf_level");
var pw=d.power_w!=null?num(d.power_w,d.power_w<10?1:0)+(d.power_limit_w?'<span class="dim">/'+num(d.power_limit_w)+'</span>':"")+" W":"–";
rows+='<tr class="dev" data-k="'+esc(key)+'"><td><span class="tag" style="background:'+color+'">'+esc(VENDORS[v]||v.slice(0,7).toUpperCase())+'</span><span class="idx">#'+esc(idx)+'</span></td>'
+'<td class="name" title="'+esc(d.name)+'">'+esc(d.name||"?")+(arch?' <span class="dim">'+esc(arch)+'</span>':"")+'</td>'
+'<td>'+bar(used,total,color)+'<b>'+fmtInt(used)+'</b><span class="dim"> / '+fmtInt(total)+' MB</span>'+(total>0?' <span class="dim">'+Math.round(used/total*100)+'%</span>':"")+'</td>'
+'<td class="num">'+(util==null?'<span class="dim">–</span>':num(util)+"%")+'</td>'
+'<td class="num '+tempCls(d.temp_c)+'">'+(d.temp_c==null?'<span class="dim">–</span>':num(d.temp_c)+"°C")+'</td>'
+'<td class="num">'+pw+'</td>'
+'<td class="num cx">'+esc(fan)+'</td>'
+'<td class="num cx">'+(clk!=null?num(clk)+(clkMax?'<span class="dim">/'+num(clkMax)+'</span>':"")+(cmem!=null?' <span class="dim">mem '+num(cmem)+'</span>':""):'<span class="dim">–</span>')+'</td>'
+'<td class="cx">'+(pst!=null?esc(pst):'<span class="dim">–</span>')+'</td>'
+'<td>'+health(d)+'</td>'
+'<td class="num">'+((d.processes||[]).length||'<span class="dim">0</span>')+'</td>'
+'<td class="cx">'+spark(hist.u,100,color)+' '+spark(hist.m,100,"var(--dim)")+'</td></tr>';
(d.mig_instances||[]).forEach(function(m){var mu=m.mem_used_mb||0,mt=m.mem_total_mb||0;rows+='<tr class="mig"><td>MIG '+esc(m.profile||m.mig_index||"")+'</td><td class="name" title="'+esc(m.uuid||"")+'">'+esc(m.uuid||m.name||"")+'</td><td>'+bar(mu,mt,color)+fmtInt(mu)+' / '+fmtInt(mt)+' MB</td><td class="num">'+(m.util_pct==null?"–":num(m.util_pct)+"%")+'</td><td></td><td></td><td class="cx"></td><td class="cx"></td><td class="cx"></td><td></td><td class="num">'+((m.processes||[]).length||"")+'</td><td class="cx"></td></tr>'});
if(open)rows+='<tr class="det"><td colspan="12">'+detail(d,v)+'</td></tr>'})});
if(!rows)html+='<div class="none">'+(Object.keys(devs).some(function(k){return (devs[k]||[]).length})?"nothing matches the filter":"no accelerators detected")+'</div>';
else html+='<table><thead><tr><th>device</th><th>name</th><th>memory</th><th class="num">util</th><th class="num">temp</th><th class="num">power</th><th class="num cx">fan</th><th class="num cx">clocks MHz</th><th class="cx">state</th><th>health</th><th class="num">procs</th><th class="cx">util · mem (tab history)</th></tr></thead><tbody>'+rows+'</tbody></table>';
html+='</section>'});
$("main").innerHTML=html||'<div class="none">no hosts</div>';
fillSelect($("fHost"),hostsSeen.map(function(x){return [x,x]}),"all hosts");
fillSelect($("fVendor"),Object.keys(vendorsSeen).sort().map(function(v){return [v,VENDORS[v]||v]}),"all vendors");
var tiles=[["hosts",S.up+(S.down?' <span class="dim">/ '+S.down+' down</span>':""),S.stale?S.stale+" stale":data.cluster?"cluster view":"this machine"],
["devices",S.dev,S.free+" free"],["device memory",S.mt>0?Math.round(S.mu/S.mt*100)+"%":"–",fmtInt(S.mu)+" / "+fmtInt(S.mt)+" MB"],
["processes",S.procs,""],["hottest",S.hot?num(S.hot.t)+"°C":"–",S.hot?S.hot.l:""],["throttled",S.thr,S.fault?S.fault+" with faults":""]];
$("summary").innerHTML=tiles.map(function(t){return '<div class="tile"><div class="k">'+t[0]+'</div><div class="v">'+t[1]+'</div><div class="s" title="'+esc(t[2])+'">'+esc(t[2])+'</div></div>'}).join("");
var when=data.updated?new Date(data.updated*1000).toLocaleTimeString():"";
$("status").textContent=(state.paused?"paused":"every "+state.interval+" s")+(when?" · updated "+when:"")+(data.cluster?" · "+hosts.length+" host(s)":"")}

function detail(d,v){var out="";var ps=d.processes||[];
if(v==="cpu"){out+='<div class="kv">'+kv("cores",(d.cores_physical!=null?d.cores_physical+" physical / ":"")+(d.cores_logical!=null?d.cores_logical+" logical":""))+kv("sockets",d.sockets)+kv("layout",d.core_layout)+kv("load",d.load_avg?d.load_avg.map(function(x){return num(x,2)}).join(" "):null)+kv("freq",d.freq_mhz!=null?num(d.freq_mhz)+(d.freq_max_mhz?" / "+num(d.freq_max_mhz):"")+" MHz":null)+kv("swap",d.swap_total_mb?fmtInt(d.swap_used_mb)+" / "+fmtInt(d.swap_total_mb)+" MB":null)+kv("numa",d.numa_nodes)+kv("cgroup cpu",d.cgroup_cpu_limit)+kv("cgroup mem",d.cgroup_mem_limit_mb!=null?fmtInt(d.cgroup_mem_limit_mb)+" MB":null)+'</div>';
if(d.per_core_pct&&d.per_core_pct.length)out+='<div class="kv" style="margin-top:4px">'+kv("per core",d.per_core_pct.map(function(x){return num(x)}).join(" "))+'</div>'}
else{out+='<div class="kv">'+kv("uuid",d.uuid)+kv("free",d.mem_free_mb!=null?fmtInt(d.mem_free_mb)+" MB":null)+kv("mem util",d.mem_util_pct!=null?num(d.mem_util_pct)+"%":null)+kv("hotspot",d.temp_hotspot_c!=null?num(d.temp_hotspot_c)+"°C":null)+kv("mem temp",d.temp_mem_c!=null?num(d.temp_mem_c)+"°C":null)+kv("enc/dec",d.enc_pct!=null||d.dec_pct!=null?num(d.enc_pct)+"% / "+num(d.dec_pct)+"%":null)+kv("ECC corr",d.ecc_corrected)+kv("ECC uncorr",d.ecc_uncorrected)+kv("throttle",(d.throttle_reasons||[]).join(", ")||null)+kv("energy",d.energy_j!=null?num(d.energy_j/1000,1)+" kJ":null)+kv("PCIe",d.pcie_gen!=null?"gen "+d.pcie_gen+" x"+(d.pcie_width||"?"):null)+kv("ANE",d.ane_power_w!=null?num(d.ane_power_w,2)+" W":null)+kv("unified",d.unified_memory===true?"yes":null)+kv("bdf",d.bdf)+kv("nvpmodel",d.nvpmodel)+'</div>'}
if(ps.length){out+='<table class="procs"><thead><tr><th>pid</th><th>user</th><th class="num">dev mem</th><th class="num">sm%</th><th class="num">rss</th><th class="num">cpu%</th><th>container / job</th><th>command</th></tr></thead><tbody>';
ps.forEach(function(p){out+='<tr><td>'+esc(p.pid)+'</td><td>'+esc(p.username||"")+'</td><td class="num">'+(p.gpu_mem_mb!=null?fmtInt(p.gpu_mem_mb)+" MB":"–")+'</td><td class="num">'+(p.sm_pct!=null?num(p.sm_pct)+"%":"–")+'</td><td class="num">'+(p.rss_mb!=null?fmtInt(p.rss_mb)+" MB":"–")+'</td><td class="num">'+(p.cpu_pct!=null?num(p.cpu_pct)+"%":"–")+'</td><td>'+esc(p.container||"")+'</td><td class="cmd" title="'+esc(p.cmdline||p.command||"")+'">'+esc(p.cmdline||p.command||"")+'</td></tr>'});out+='</tbody></table>'}
else out+='<div class="dim" style="margin-top:4px">no processes reported</div>';
return out}
function kv(k,v){return (v==null||v==="")?"":'<span>'+esc(k)+' <b>'+esc(v)+'</b></span>'}
function fillSelect(sel,items,label){var cur=sel.value;var h='<option value="">'+label+'</option>';items.forEach(function(it){h+='<option value="'+esc(it[0])+'"'+(it[0]===cur?" selected":"")+'>'+esc(it[1])+'</option>'});if(sel.innerHTML!==h)sel.innerHTML=h;sel.value=cur}

function poll(){if(state.paused)return;var x=new XMLHttpRequest();x.open("GET","cluster?t="+Date.now(),true);x.timeout=60000;
x.onload=function(){try{var d=JSON.parse(x.responseText);if(d.error){$("status").textContent="server error: "+d.error;return}state.data=d;state.interval=d.interval||state.interval;state.errors=0;if(d.version){$("ver").textContent=d.version;document.title="xpustat · "+(d.cluster?(d.hosts||[]).length+" hosts":((d.hosts||[])[0]||{}).host||"")}render()}catch(e){$("status").textContent="bad response: "+e}schedule()};
x.onerror=x.ontimeout=function(){state.errors++;$("status").textContent="server unreachable (retry "+state.errors+")";schedule()};x.send()}
function schedule(){clearTimeout(state.timer);state.timer=setTimeout(poll,Math.max(500,state.interval*1000))}
$("main").addEventListener("click",function(e){var tr=e.target.closest("tr.dev");if(!tr)return;if(window.getSelection&&String(window.getSelection()))return;var k=tr.getAttribute("data-k");state.expanded[k]=!state.expanded[k];render()});
$("expandAll").addEventListener("click",function(){var any=Object.keys(state.expanded).some(function(k){return state.expanded[k]});state.expanded={};if(!any)document.querySelectorAll("tr.dev").forEach(function(tr){state.expanded[tr.getAttribute("data-k")]=true});$("expandAll").textContent=any?"expand all":"collapse all";render()});
$("pause").addEventListener("click",function(){state.paused=!state.paused;$("pause").textContent=state.paused?"resume":"pause";$("pause").classList.toggle("on",state.paused);if(!state.paused)poll();else render()});
["fHost","fVendor"].forEach(function(id){$(id).addEventListener("change",render)});$("fText").addEventListener("input",render);
$("compact").addEventListener("change",function(){document.body.classList.toggle("compact",this.checked);try{localStorage.setItem("xpustat.compact",this.checked?"1":"")}catch(e){}});
try{if(localStorage.getItem("xpustat.compact")){$("compact").checked=true;document.body.classList.add("compact")}}catch(e){}
document.addEventListener("visibilitychange",function(){if(!document.hidden&&!state.paused)poll()});
poll();
})();
</script>
</body>
</html>
"""
