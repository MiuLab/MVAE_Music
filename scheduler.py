import math
import random

class Scheuduler(object):

    def get_value(self, t):
        raise NotImplementedError

    def sample(self, t):
        return random.random() < self.get_value(t)

class InverseSigmoidScheduler(Scheuduler):
    def __init__(self, start, end, k):
        self.start = start
        self.end = end
        self.k = k

    def get_value(self, t):
        return self.end + (self.start - self.end) * self.k / (self.k + math.exp(t / self.k))

class LinearScheduler(Scheuduler):
    def __init__(self, start, end, k):
        self.start = start
        self.end = end
        self.k = k

    def get_value(self, t):
        return max(self.end, (self.start - self.k * t))

class InverseLinearScheduler(LinearScheduler):
    def __init__(self, start, end, k):
        super(InverseLinearScheduler, self).__init__(start, end, k)

    def get_value(self, t):
        return 1 - super(InverseLinearScheduler, self).get_value(t)

class ConstantScheduler(Scheuduler):
    def __init__(self, start, end, k):
        self.start = start
        self.end = end
        self.k = k

    def get_value(self, t):
        return self.k

class LogisticScheduler(Scheuduler):
    def __init__(self, start, end, k, x0):
        self.start = float(start)
        self.end = float(end)
        self.k = float(k)
        self.x0 = float(x0)

    def get_value(self, t):
        return (self.end - self.start)/(1+math.exp(-self.k * (t - self.x0))) \
                + self.start

if __name__ == '__main__':
    #s = InverseLinearScheduler(1.0, 0.0, 0.005)
    s = LogisticScheduler(0.0, 1.0, 0.1, 100)
    for t in range(200):
        print(t, s.get_value(t))
