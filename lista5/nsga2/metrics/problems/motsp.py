from metrics.hvr import HVR, HV    
    
class MOTSPMetrics():
    def HV(self, front):
        return HV([0, 0])(front)

    def HVR(self, front):
        return HVR([0, 0], 0)(front)
