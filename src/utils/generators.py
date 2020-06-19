import threading
import numpy as np
from queue import Queue


def sampling_generator(data):
    len_data = len(data)
    while True:
        if len_data > 1:
            yield data[np.random.choice(len(data))]
        elif len_data == 1:
            yield data[0]
        else:
            print(data)
            raise Exception('No data available for sampling generator')


class BatchGenerator(threading.Thread):
    def __init__(self, generator, batch_size, batcher):
        threading.Thread.__init__(self)
        self.queue = Queue(4)
        self.generator = generator
        self.batch_size = batch_size
        self.batcher = batcher
        self.daemon = True
        self.start()

    def run(self):
        batch_x = []
        item = next(self.generator)
        while item is not None:
            batch_x.append(item)
            if len(batch_x) == self.batch_size:
                self.queue.put(self.batcher(batch_x))
                batch_x = []
            item = next(self.generator, None)
        self.queue.put(None)

    def __iter__(self):
        return self

    def __next__(self):
            # print(f'BatchGenerator - {id(self)} - {self.queue.qsize()} / 4')
            next_item = self.queue.get()
            if next_item is None:
                raise StopIteration
            return next_item

