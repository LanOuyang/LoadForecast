import matplotlib.pyplot as plt
import numpy as np

plt.ion()

class DynamicUpdate():

    def __init__(self, num_lines=3, limit=2000):
        self.figure, self.ax = plt.subplots(figsize=(10,6))
        self.lines = self.ax.plot(np.empty((0, num_lines)), np.empty((0, num_lines)))
        self.ax.legend(self.lines, ("prediction", "target", "MAE"))

        self.limit = limit

        # Autoscale on unknown axis and known lims on the other
        self.ax.set_autoscaley_on(True)

    def __call__(self, xdata, ydata):
        # Update data (with the new _and_ the old points)

        #print(xdata.shape, ydata.shape)
        
        for line_k, y_k in zip(self.lines, ydata.T):
                #print(y_k.shape)
                if self.limit:
                    line_k.set_xdata(np.append(line_k.get_xdata(), xdata)[-self.limit:])
                    line_k.set_ydata(np.append(line_k.get_ydata(), y_k)[-self.limit:])
                else:
                    line_k.set_xdata(np.append(line_k.get_xdata(), xdata))
                    line_k.set_ydata(np.append(line_k.get_ydata(), y_k))

        # Need both of these in order to rescale
        self.ax.relim()
        self.ax.autoscale_view()

        # We need to draw *and* flush
        self.figure.canvas.draw()
        self.figure.canvas.flush_events()