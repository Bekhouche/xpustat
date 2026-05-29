import amdsmi

amdsmi.amdsmi_init()
handles = amdsmi.amdsmi_get_processor_handles()
for i, handle in enumerate(handles):
    print(f"GPU {i}")
    try:
        procs = amdsmi.amdsmi_get_gpu_process_list(handle)
        print("Procs list:", procs)
        for pid in procs:
            info = amdsmi.amdsmi_get_gpu_compute_process_info_by_pid(pid)
            print("Info:", info)
    except Exception as e:
        print("Error getting procs:", e)
amdsmi.amdsmi_shut_down()
