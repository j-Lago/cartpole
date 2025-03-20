import time
class TicToc():
    def __init__(self):
        self.start = None
        self.elapsed = None

    def tic(self) -> time:
        self.start = time.time()
        return self.start

    def toc(self) -> time:
        end = time.time()
        self.elapsed = end - self.start
        return self.elapsed

    def __str__(self):
        return str(self.elapsed) + 's'

    def __format__(self, format_spec):
        if format_spec:
            return format(self.elapsed, format_spec) + 's'
        return str(self.elapsed) + 's'


tictoc = TicToc()
