from metrics.hvr import HVR, HV    
    
class MOTSPMetricsAB():
    def HV(self, front):
        return HV([200000, 200000])(front)

    def HVR(self, front):
        return HVR([200000, 200000], 17035431968)(front)

class MOTSPMetricsCD():
    def HV(self, front):
        return HV([200000, 200000])(front)

    def HVR(self, front):
        return HVR([200000, 200000], )(front)
    
class MOTSPMetricsEF():
    def HV(self, front):
        return HV([200000, 200000])(front)

    def HVR(self, front):
        return HVR([200000, 200000], )(front)
