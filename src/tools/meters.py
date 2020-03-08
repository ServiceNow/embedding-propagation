class BasicMeter(object):
    """
    Basic class to monitor scores
    """
    meters = {}
    submeters = {}

    @staticmethod
    def get(name, recursive=False, tag=None, force=True):
        """ Creates a new meter or returns an already existing one with the given name.

        Args:
            name: meter name

        Returns: BasicMeter instance

        """

        if name not in BasicMeter.meters:
            if recursive:
                for supername, meter in BasicMeter.meters.items():
                    for subname in meter.submeters:
                        if "%s_%s" %(supername, subname) == name:
                            return meter.submeters[subname]
            if force:
                BasicMeter.meters[name] = BasicMeter(name)
            else:
                raise ModuleNotFoundError

        if tag is not None:
            if force:
                return BasicMeter.meters[name].get_submeter(tag)
            else:
                raise ModuleNotFoundError

        return BasicMeter.meters[name]

    @staticmethod


    @staticmethod
    def dict():
        """ Obtain meters in a dictionary

        Returns: dictionary of BasicMeter

        """
        return BasicMeter.meters

    def __init__(self, name=""):
        """
        Constructor
        """
        self.count = 0.
        self.total = 0.
        self.name = name
        self.submeters = {}

    def get_submeter(self, name):
        if name not in self.submeters:
            name_ = "%s_%s" %(self.name, name)
            self.submeters[name] = BasicMeter(name_)
        return self.submeters[name]

    def update(self, v, count, tag=None):
        """ Update meter values

        Args:
            v: current value
            count: N if value is the average of N values.

        Returns: self

        """
        self.count += count
        self.total += v

        if tag is not None:
            if not isinstance(tag, list):
                tag = [tag]
            for t in tag:
                self.get_submeter(t).update(v, count)
        return self

    def mean(self, tag=None, recursive=False):
        """ Computes the mean of the current values

        Returns: mean of the current values (float)

        """
        if recursive:
            try:
                ret = { self.name: self.total / self.count }
            except ZeroDivisionError:
                return { self.name: 0 }
            for submeter in self.submeters:
                ret.update(self.get_submeter(submeter).mean(recursive=True))
            return ret
        if tag is not None:
            return self.get_submeter(tag).mean()
        else:
            return self.total / self.count

    def reset(self):
        """ Resets the meter.

        Returns: self

        """
        for submeter in self.submeters:
            self.submeters[submeter].reset()
        self.count = 0
        self.total = 0

        return self

