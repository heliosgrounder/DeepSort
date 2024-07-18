class FPSMetric():
    def __init__(self, vis):
        self.__vis = vis
    
    def get_metric(self):
        try:
            return self.__vis.get_fps()
        except Exception:
            return None