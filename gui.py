import sys
import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as Axes3D

class GUI():
    def __init__(self, span):
        self.span = span
        self.fig = plt.figure()
        self.ax = Axes3D.Axes3D(self.fig)
        self.ax.set_xlim3d([-10., 10.])
        self.ax.set_xlabel('X')
        self.ax.set_ylim3d([-10., 10.])
        self.ax.set_ylabel('Y')
        self.ax.set_zlim3d([-5., 15.])
        self.ax.set_zlabel('Z')
        self.ax.set_title('Quadcopter Simulation')

        l1 = self.ax.plot([], [], [], color='blue', linewidth=3, antialiased=False)[0]
        l2 = self.ax.plot([], [], [], color='red', linewidth=3, antialiased=False)[0]
        normal = self.ax.plot([], [], [], color='black', linewidth=2, antialiased=False)[0]
        hub = self.ax.plot([], [], [], marker='o', color='green', markersize=6, antialiased=False)[0]
        self.quadrotors = [{'l1': l1, 'l2': l2, 'normal': normal, 'hub': hub}]

        self.follow = False
        self.fig.canvas.mpl_connect('key_press_event', self.keypress)

    def add_quadrotor(self):
        l1 = self.ax.plot([], [], [], color='blue', linewidth=3, antialiased=False)[0]
        l2 = self.ax.plot([], [], [], color='red', linewidth=3, antialiased=False)[0]
        normal = self.ax.plot([], [], [], color='black', linewidth=2, antialiased=False)[0]
        hub = self.ax.plot([], [], [], marker='o', color='green', markersize=6, antialiased=False)[0]
        self.quadrotors.append({'l1': l1, 'l2': l2, 'normal': normal, 'hub': hub})

    def update_batch(self, states, indices):
        for i in indices:
            self.update(states[i], i)

    def update(self, state, idx):
        R = state['rotation_matrix']
        span = self.span

        points = np.array([[-span, 0, 0], [span, 0, 0], [0, -span, 0], [0, span, 0], [0, 0, span/2.], [0, 0, 0]]).T
        #points = np.dot(np.array([[0.70710678, -0.70710678, 0.],
        #                       [0.70710678,  0.70710678, 0.],
        #                       [0.        ,  0.        , 1.]]), points)
        points = np.dot(R, points)

        points[0, :] += state['position'][0]
        points[1, :] += state['position'][1]
        points[2, :] += state['position'][2]

        self.quadrotors[idx]['l1'].set_data(points[0,0:2],points[1,0:2])
        self.quadrotors[idx]['l1'].set_3d_properties(points[2,0:2])
        self.quadrotors[idx]['l2'].set_data(points[0,2:4],points[1,2:4])
        self.quadrotors[idx]['l2'].set_3d_properties(points[2,2:4])
        self.quadrotors[idx]['normal'].set_data(points[0,4:6],points[1,4:6])
        self.quadrotors[idx]['normal'].set_3d_properties(points[2,4:6])
        self.quadrotors[idx]['hub'].set_data(points[0,5],points[1,5])
        self.quadrotors[idx]['hub'].set_3d_properties(points[2,5])

        if self.follow:
            x = [points[0, 5]-10, points[0, 5]+10]
            y = [points[1, 5]-10, points[1, 5]+10]
            z = [points[2, 5]-10, points[2, 5]+10]
            self.ax.set_xlim3d(x)
            self.ax.set_ylim3d(y)
            self.ax.set_zlim3d(z)

        plt.pause(0.000000000000001)

    def keypress(self,event):
        sys.stdout.flush()
        if event.key == 'e':
            self.follow = not self.follow
        elif event.key == 'x':
            y = list(self.ax.get_ylim3d())
            y[0] += 1
            y[1] += 1
            self.ax.set_ylim3d(y)
        elif event.key == 'w':
            y = list(self.ax.get_ylim3d())
            y[0] -= 1
            y[1] -= 1
            self.ax.set_ylim3d(y)
        elif event.key == 'd':
            x = list(self.ax.get_xlim3d())
            x[0] += 1
            x[1] += 1
            self.ax.set_xlim3d(x)
        elif event.key == 'a':
            x = list(self.ax.get_xlim3d())
            x[0] -= 1
            x[1] -= 1
            self.ax.set_xlim3d(x)
