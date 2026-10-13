"""Create a Vulkan device on the UUID of physical GPU 0 only."""
import ctypes as c
import json
from pathlib import Path
import uuid

class ApplicationInfo(c.Structure):
    _fields_ = [('sType', c.c_uint32), ('pNext', c.c_void_p), ('pApplicationName', c.c_char_p),
                ('applicationVersion', c.c_uint32), ('pEngineName', c.c_char_p),
                ('engineVersion', c.c_uint32), ('apiVersion', c.c_uint32)]
class InstanceInfo(c.Structure):
    _fields_ = [('sType', c.c_uint32), ('pNext', c.c_void_p), ('flags', c.c_uint32),
                ('pApplicationInfo', c.POINTER(ApplicationInfo)), ('enabledLayerCount', c.c_uint32),
                ('ppEnabledLayerNames', c.c_void_p), ('enabledExtensionCount', c.c_uint32),
                ('ppEnabledExtensionNames', c.c_void_p)]
class DeviceID(c.Structure):
    _fields_ = [('sType', c.c_uint32), ('pNext', c.c_void_p), ('deviceUUID', c.c_ubyte * 16),
                ('driverUUID', c.c_ubyte * 16), ('deviceLUID', c.c_ubyte * 8),
                ('deviceNodeMask', c.c_uint32), ('deviceLUIDValid', c.c_uint32)]
class Properties(c.Structure):
    _fields_ = [('sType', c.c_uint32), ('pNext', c.c_void_p), ('properties', c.c_ubyte * 1024)]
class QueueProperties(c.Structure):
    _fields_ = [('queueFlags', c.c_uint32), ('queueCount', c.c_uint32),
                ('timestampValidBits', c.c_uint32), ('granularity', c.c_uint32 * 3)]
class QueueInfo(c.Structure):
    _fields_ = [('sType', c.c_uint32), ('pNext', c.c_void_p), ('flags', c.c_uint32),
                ('queueFamilyIndex', c.c_uint32), ('queueCount', c.c_uint32),
                ('pQueuePriorities', c.POINTER(c.c_float))]
class DeviceInfo(c.Structure):
    _fields_ = [('sType', c.c_uint32), ('pNext', c.c_void_p), ('flags', c.c_uint32),
                ('queueCreateInfoCount', c.c_uint32), ('pQueueCreateInfos', c.POINTER(QueueInfo)),
                ('enabledLayerCount', c.c_uint32), ('ppEnabledLayerNames', c.c_void_p),
                ('enabledExtensionCount', c.c_uint32), ('ppEnabledExtensionNames', c.c_void_p),
                ('pEnabledFeatures', c.c_void_p)]

vk = c.CDLL('libvulkan.so.1')
handle = c.c_void_p
vk.vkCreateInstance.argtypes = [c.POINTER(InstanceInfo), c.c_void_p, c.POINTER(handle)]
vk.vkEnumeratePhysicalDevices.argtypes = [handle, c.POINTER(c.c_uint32), c.POINTER(handle)]
vk.vkGetPhysicalDeviceProperties2.argtypes = [handle, c.POINTER(Properties)]
vk.vkGetPhysicalDeviceQueueFamilyProperties.argtypes = [handle, c.POINTER(c.c_uint32), c.POINTER(QueueProperties)]
vk.vkCreateDevice.argtypes = [handle, c.POINTER(DeviceInfo), c.c_void_p, c.POINTER(handle)]
vk.vkDestroyDevice.argtypes = [handle, c.c_void_p]
vk.vkDestroyInstance.argtypes = [handle, c.c_void_p]
app = ApplicationInfo(sType=0, pApplicationName=b'RoboTwin GPU 0 probe', apiVersion=(1 << 22) | (1 << 12))
instance_info = InstanceInfo(sType=1, pApplicationInfo=c.pointer(app))
instance = handle()
status = vk.vkCreateInstance(c.byref(instance_info), None, c.byref(instance))
assert status == 0, ('vkCreateInstance', status)
device = handle()
try:
    count = c.c_uint32()
    assert vk.vkEnumeratePhysicalDevices(instance, c.byref(count), None) == 0
    physical = (handle * count.value)()
    assert vk.vkEnumeratePhysicalDevices(instance, c.byref(count), physical) == 0
    expected = uuid.UUID('eec5c276-20b4-ff41-4459-a2dee5affb92')
    selected = None
    for gpu in physical:
        ids = DeviceID(sType=1000071004)
        props = Properties(sType=1000059001, pNext=c.cast(c.pointer(ids), c.c_void_p))
        vk.vkGetPhysicalDeviceProperties2(gpu, c.byref(props))
        if bytes(ids.deviceUUID) == expected.bytes:
            selected = gpu
            gpu_name = bytes(props.properties[20:276]).split(b'\x00')[0].decode()
            break
    assert selected is not None, 'Physical GPU 0 UUID not present in Vulkan'
    qcount = c.c_uint32()
    vk.vkGetPhysicalDeviceQueueFamilyProperties(selected, c.byref(qcount), None)
    queues = (QueueProperties * qcount.value)()
    vk.vkGetPhysicalDeviceQueueFamilyProperties(selected, c.byref(qcount), queues)
    family = next(i for i, q in enumerate(queues) if q.queueFlags & 1 and q.queueCount)
    priority = c.c_float(1.0)
    queue_info = QueueInfo(sType=2, queueFamilyIndex=family, queueCount=1, pQueuePriorities=c.pointer(priority))
    device_info = DeviceInfo(sType=3, queueCreateInfoCount=1, pQueueCreateInfos=c.pointer(queue_info))
    status = vk.vkCreateDevice(selected, c.byref(device_info), None, c.byref(device))
    assert status == 0, ('vkCreateDevice on physical GPU 0', status)
    result = {'passed': True, 'gpu_index': 0, 'uuid': str(expected), 'gpu_name': gpu_name,
              'graphics_queue_family': family, 'logical_devices_created': 1}
    Path(__file__).with_name('vulkan-probe.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
finally:
    if device:
        vk.vkDestroyDevice(device, None)
    vk.vkDestroyInstance(instance, None)
