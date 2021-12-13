

from PyQt5 import QtCore, QtWidgets
import sys
from multiprocessing import Process, Queue
from time import time
import vtk
from vtk.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
import numpy as np
from time import sleep

# Some other functions I toyed with before I got the vtk example working
def add_numbers(num1, num2):
    return num1 + num2


def numpy_arange(num1, num2):
    import numpy as np
    return np.arange(num1, num2)


def worker(inputs_queue, outputs_queue, proc_id):
    import vtk
    print(f'Initializing worker {proc_id}')
    while True:
        sleep(0.01)
        if not inputs_queue.empty():
            print(f'[{proc_id}] Receiving message')
            message = inputs_queue.get()
            print(f'[{proc_id}] message:', message)

            if message == 'STOP':
                print(f'[{proc_id}] stopping')
                break
            else:
                print(f'[{proc_id}] computing')
                t_work1 = time()
                num1, num2, meta = message

                apd = vtk.vtkAppendPolyData()

                for _ in range(int(num1)):
                    x, y, z = np.random.rand(3)
                    sphere = vtk.vtkSphereSource()
                    sphere.SetRadius(0.1)
                    sphere.SetCenter(x, y, z)
                    sphere.Update()
                    apd.AddInputData(sphere.GetOutput())

                apd.Update()
                w = vtk.vtkPolyDataWriter()
                w.WriteToOutputStringOn()
                w.SetInputData(apd.GetOutput())
                w.Update()
                sphere_mesh = w.GetOutputString()
                print(f'[{proc_id}] Result size: {sys.getsizeof(sphere_mesh)}')

                t2 = time()
                meta['t2'] = t2
                meta['t_work'] = t2 - t_work1
                meta['proc_id'] = proc_id
                r_message = (sphere_mesh, meta)
                outputs_queue.put(r_message)


class VTKPlot(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()

        self.frame = QtWidgets.QFrame()

        self.vl = QtWidgets.QVBoxLayout()
        self.vtkWidget = QVTKRenderWindowInteractor(self.frame)
        self.vl.addWidget(self.vtkWidget)

        self.ren = vtk.vtkRenderer()
        self.vtkWidget.GetRenderWindow().AddRenderer(self.ren)
        self.renwin = self.vtkWidget.GetRenderWindow()
        self.iren = self.renwin.GetInteractor()

        # Depth Peeling
        self.renwin.SetAlphaBitPlanes(1)
        self.renwin.SetMultiSamples(0)
        self.ren.SetUseDepthPeeling(1)
        self.ren.SetMaximumNumberOfPeels(100)
        self.ren.SetOcclusionRatio(0.1)

        self.ren.ResetCamera()

        self.frame.setLayout(self.vl)
        self.setCentralWidget(self.frame)

        self.iren.SetInteractorStyle(vtk.vtkInteractorStyleTrackballCamera())
        self.iren.Initialize()


class DemoGui(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()

        self._layout = QtWidgets.QVBoxLayout()
        self._number1 = QtWidgets.QLineEdit('5')
        self._number1.setToolTip('Number of Spheres per job')
        self._number2 = QtWidgets.QLineEdit('4')
        self._number2.setToolTip('Number of jobs to generate')
        self._opacity = QtWidgets.QSlider()
        self._opacity.setToolTip('Opacity of Spheres. Evaluated just before render, after polydata has returned from'
                                 ' the worker. Renderer gets pretty bogged down with opacity and 1000+ spheres.')
        self._opacity.setMinimum(0)
        self._opacity.setMaximum(100)
        self._opacity.setValue(100)
        self._send_button = QtWidgets.QPushButton('send')
        self._stop_button = QtWidgets.QPushButton('stop worker')
        self._clear_button = QtWidgets.QPushButton('clear outputs')

        self._textbox = QtWidgets.QTextEdit()
        self._vtk_plot = VTKPlot()

        self._inputs_queue = Queue()
        self._outputs_queue = Queue()

        self._worker_process1 = Process(target=worker, args=(self._inputs_queue, self._outputs_queue, 1))
        self._worker_process2 = Process(target=worker, args=(self._inputs_queue, self._outputs_queue, 2))
        self._worker_process3 = Process(target=worker, args=(self._inputs_queue, self._outputs_queue, 3))
        self._worker_process4 = Process(target=worker, args=(self._inputs_queue, self._outputs_queue, 4))
        self._worker_process1.start()
        self._worker_process2.start()
        self._worker_process3.start()
        self._worker_process4.start()

        self._timer = QtCore.QTimer()
        self._timer.setInterval(100)
        self._timer.timeout.connect(self.display_results)
        self._timer.start()

        self._send_button.clicked.connect(self.send_numbers)
        self._stop_button.clicked.connect(self.stop)
        self._clear_button.clicked.connect(self.clear)

        self._layout.addWidget(self._number1)
        self._layout.addWidget(self._number2)
        self._layout.addWidget(self._opacity)
        self._layout.addWidget(self._send_button)
        self._layout.addWidget(self._stop_button)
        self._layout.addWidget(self._clear_button)
        self._layout.addWidget(self._textbox)
        self._layout.addWidget(self._vtk_plot)
        self.setLayout(self._layout)

        self._time_dict = {}

    def closeEvent(self, event):
        self.stop()

    def clear(self, *args):
        self._textbox.clear()
        self._vtk_plot.ren.Clear()
        for actor in self._vtk_plot.ren.GetActors():
            self._vtk_plot.ren.RemoveActor(actor)

        self._vtk_plot.ren.ResetCamera()
        self._vtk_plot.ren.Render()
        self._vtk_plot.renwin.Render()

    def stop(self, *args):
        print('sending stop')
        for _ in range(4):
            self._inputs_queue.put('STOP')

    def send_numbers(self, *args):
        print('sending numbers')

        try:
            num1 = float(self._number1.text())
            num2 = float(self._number2.text())
        except Exception as e:
            import traceback
            traceback.print_exc()
            return

        meta = {'t1': time()}
        for _ in range(int(num2)):
            self._inputs_queue.put((num1, num2, meta))

    def display_results(self):
        if not self._outputs_queue.empty():
            # Not sure if this is necessary, but I don't want to think about what happens when the timer goes off while
            # the method is in progress.
            self._timer.blockSignals(True)
            t0 = time()
            n = self._outputs_queue.qsize()
            # This is a compromize... I would keep emptying the queue until it's empty, which is more likely to lock
            #  the GUI because the workers may continue catching up while this is operating.
            # If there is no loop here, then items are only read once per timer tick.
            # If the worker puts each sphere in the queue individually, this can cause each sphere's appearance to
            #  be animated which is much slower because Render() is slow... however workers are currently appending all spheres
            #  of a job into an AppendPolyData
            for _ in range(n):
                # print('Displaying results')
                r_message = self._outputs_queue.get()
                result, meta = r_message
                t1 = meta['t1']
                t_work = meta['t_work']
                t3 = time()
                dt = t3-t1
                t_overhead = dt - t_work
                proc_id = meta['proc_id']
                # self._textbox.append(str(result))
                self._textbox.append(f'Took {dt:1.6f} s total. {t_work:1.6f} working. {t_overhead:1.6f} overhead. Proc: {proc_id}')
                self.plot_shape(result, proc_id)
            t1 = time()
            self._timer.blockSignals(False)
            print(f'Output Processing Time: {t1-t0:1.3f}s')
            self._vtk_plot.ren.ResetCamera()
            self._vtk_plot.ren.Render()
            self._vtk_plot.renwin.Render()

    def plot_shape(self, shape_data, proc_id):

        t0 = time()
        reader = vtk.vtkPolyDataReader()
        reader.ReadFromInputStringOn()
        reader.SetInputString(shape_data)
        reader.Update()
        shape_pd = reader.GetOutput()

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(shape_pd)
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        color = {1: (1, 0, 0),
                 2: (0, 1, 0),
                 3: (0, 0, 1),
                 4: (0, 1, 1)}[proc_id]
        actor.GetProperty().SetDiffuseColor(color)
        actor.GetProperty().SetOpacity(self._opacity.value()/100)

        self._vtk_plot.ren.AddActor(actor)

        t1 = time()
        # print(f'Render Time: {t1-t0:1.3f}s')



if __name__ == '__main__':

    app = QtWidgets.QApplication([])
    window = DemoGui()
    window.show()
    app.exec_()

