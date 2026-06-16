import ctypes
import os
from ctypes import c_float, c_int32, c_wchar, wintypes


AC_STATUS = c_int32
AC_SESSION_TYPE = c_int32
AC_FLAG_TYPE = c_int32


class SPageFilePhysics(ctypes.Structure):
    _pack_ = 4
    _fields_ = [
        ("packetId", c_int32),
        ("gas", c_float),
        ("brake", c_float),
        ("fuel", c_float),
        ("gear", c_int32),
        ("rpms", c_int32),
        ("steerAngle", c_float),
        ("speedKmh", c_float),
        ("velocity", c_float * 3),
        ("accG", c_float * 3),
        ("wheelSlip", c_float * 4),
        ("wheelLoad", c_float * 4),
        ("wheelsPressure", c_float * 4),
        ("wheelAngularSpeed", c_float * 4),
        ("tyreWear", c_float * 4),
        ("tyreDirtyLevel", c_float * 4),
        ("tyreCoreTemperature", c_float * 4),
        ("camberRAD", c_float * 4),
        ("suspensionTravel", c_float * 4),
        ("drs", c_float),
        ("tc", c_float),
        ("heading", c_float),
        ("pitch", c_float),
        ("roll", c_float),
        ("cgHeight", c_float),
        ("carDamage", c_float * 5),
        ("numberOfTyresOut", c_int32),
        ("pitLimiterOn", c_int32),
        ("abs", c_float),
        ("kersCharge", c_float),
        ("kersInput", c_float),
        ("autoShifterOn", c_int32),
        ("rideHeight", c_float * 2),
        ("turboBoost", c_float),
        ("ballast", c_float),
        ("airDensity", c_float),
        ("airTemp", c_float),
        ("roadTemp", c_float),
        ("localAngularVel", c_float * 3),
        ("finalFF", c_float),
        ("brakeTemp", c_float * 4),
        ("clutch", c_float),
        ("tyreTempI", c_float * 4),
        ("tyreTempM", c_float * 4),
        ("tyreTempO", c_float * 4),
        ("isAIControlled", c_int32),
        ("tyreContactPoint", (c_float * 4) * 3),
        ("tyreContactNormal", (c_float * 4) * 3),
        ("tyreContactHeading", (c_float * 3) * 4),
        ("brakeBias", c_float),
        ("localVelocity", c_float * 3),
    ]


class SPageFileGraphic(ctypes.Structure):
    _pack_ = 4
    _fields_ = [
        ("packetId", c_int32),
        ("status", AC_STATUS),
        ("session", AC_SESSION_TYPE),
        ("currentTime", c_wchar * 15),
        ("lastTime", c_wchar * 15),
        ("bestTime", c_wchar * 15),
        ("split", c_wchar * 15),
        ("completedLaps", c_int32),
        ("position", c_int32),
        ("iCurrentTime", c_int32),
        ("iLastTime", c_int32),
        ("iBestTime", c_int32),
        ("sessionTimeLeft", c_float),
        ("distanceTraveled", c_float),
        ("isInPit", c_int32),
        ("currentSectorIndex", c_int32),
        ("lastSectorTime", c_int32),
        ("numberOfLaps", c_int32),
        ("tyreCompound", c_wchar * 33),
        ("replayTimeMultiplier", c_float),
        ("normalizedCarPosition", c_float),
        ("carCoordinates", c_float * 3),
        ("penaltyTime", c_float),
        ("flag", AC_FLAG_TYPE),
        ("idealLineOn", c_int32),
        ("isInPitLane", c_int32),
        ("surfaceGrip", c_float),
        ("mandatoryPitDone", c_int32),
        ("windSpeed", c_float),
        ("windDirection", c_float),
    ]


class _WindowsSharedMemoryStruct:
    def __init__(self, names, structure_type):
        self._names = names
        self._structure_type = structure_type
        self._size = ctypes.sizeof(structure_type)
        self._handle = None
        self._view = None
        self._page = None

        self._kernel32 = None
        if os.name == "nt":
            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            kernel32.OpenFileMappingW.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.LPCWSTR]
            kernel32.OpenFileMappingW.restype = wintypes.HANDLE
            kernel32.MapViewOfFile.argtypes = [
                wintypes.HANDLE,
                wintypes.DWORD,
                wintypes.DWORD,
                wintypes.DWORD,
                ctypes.c_size_t,
            ]
            kernel32.MapViewOfFile.restype = wintypes.LPVOID
            kernel32.UnmapViewOfFile.argtypes = [wintypes.LPCVOID]
            kernel32.UnmapViewOfFile.restype = wintypes.BOOL
            kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
            kernel32.CloseHandle.restype = wintypes.BOOL
            self._kernel32 = kernel32

    def read(self):
        if self._page is None and not self._connect():
            return None
        return self._page

    def close(self):
        if self._kernel32 is not None and self._view:
            self._kernel32.UnmapViewOfFile(self._view)
        if self._kernel32 is not None and self._handle:
            self._kernel32.CloseHandle(self._handle)
        self._view = None
        self._handle = None
        self._page = None

    def _connect(self):
        if self._kernel32 is None:
            return False

        file_map_read = 0x0004
        for name in self._names:
            handle = self._kernel32.OpenFileMappingW(file_map_read, False, name)
            if not handle:
                continue

            view = self._kernel32.MapViewOfFile(handle, file_map_read, 0, 0, self._size)
            if not view:
                self._kernel32.CloseHandle(handle)
                continue

            self._handle = handle
            self._view = view
            self._page = self._structure_type.from_address(view)
            return True

        return False

    def __del__(self):
        self.close()


_physics = _WindowsSharedMemoryStruct(("acpmf_physics", "Local\\acpmf_physics"), SPageFilePhysics)
_graphics = _WindowsSharedMemoryStruct(("acpmf_graphics", "Local\\acpmf_graphics"), SPageFileGraphic)


def read_physics_page():
    return _physics.read()


def read_graphics_page():
    return _graphics.read()

