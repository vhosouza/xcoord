import threading, queue
import dti_funcs as dti
import os, psutil, vtk
import numpy as np


data_dir = os.environ.get('OneDriveConsumer') + '\\data\\dti'
FOD_path = 'sub-P0_dwi_FOD.nii'
filename = os.path.join(data_dir, FOD_path)
params = {'seed_max': 1, 'step_size': 0.1, 'min_fod': 0.1, 'probe_quality': 3,
          'max_interval': 5, 'min_radius_curv': 0.8, 'probe_length': 0.4,
          'write_interval': 10, 'numb_threads': 2 * psutil.cpu_count()}
trekker = dti.start_trekker(filename, params)
n_tracts = 2 * psutil.cpu_count()
# np.repeat(seed_trk, n_threads, axis=0)

class QueueCustom(queue.Queue):
    """
    A custom queue subclass that provides a :meth:`clear` method.
    https://stackoverflow.com/questions/6517953/clear-all-items-from-the-queue
    """

    def clear(self):
        """
        Clears all items from the queue.
        """

        with self.mutex:
            unfinished = self.unfinished_tasks - len(self.queue)
            if unfinished <= 0:
                if unfinished < 0:
                    raise ValueError('task_done() called too many times')
                self.all_tasks_done.notify_all()
            self.unfinished_tasks = unfinished
            self.queue.clear()
            self.not_full.notify_all()

###########################################################################################

counter = 0

# counter_queue = queue.Queue()
counter_queue = QueueCustom()


def counter_manager():
    'I have EXCLUSIVE rights to update the counter variable'
    global counter

    while True:
        increment = counter_queue.get()
        counter += increment[0]
        coord = increment[1]
        # print_queue.put([
        #     'The count is %d' % counter,
        #     '---------------'])
        print_queue.put([counter, coord])
        counter_queue.task_done()


t = threading.Thread(target=counter_manager)
t.daemon = True
t.start()
del t

###########################################################################################

# print_queue = queue.Queue()
print_queue = QueueCustom()

def print_manager():
    'I have EXCLUSIVE rights to call the "print" keyword'
    while True:
        job = print_queue.get()
        # for line in job:
        root = vtk.vtkMultiBlockDataSet()
        if len(job[1]) > 0:
            trekker.seed_coordinates(np.repeat(job[1], n_tracts, axis=0))
            trk_list = trekker.run()
            root = dti.tracts_computation(trk_list, root, 0)
            print('The count is {} for {} tracts'.format(job[0], len(trk_list)))
        else:
            print('The count is {} and job 1 {}'.format(job[0], job[1]))

        print_queue.task_done()


t = threading.Thread(target=print_manager)
t.daemon = True
t.start()
del t

###########################################################################################

print_queue.put(['Starting up', []])


def worker():
    'My job is to increment the counter and print the current count'
    while True:
        coord = np.array([[-8.49, -8.39, 2.5]])
        counter_queue.put([1, coord])


t = threading.Thread(target=worker)
t.daemon = True
t.start()
del t

count_aux = 0
while True:
    count_aux += 1
    if counter == 7:
        counter_queue.join()
        print_queue.put(['Finishing up', []])
        print_queue.join()
