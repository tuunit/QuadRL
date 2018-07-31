import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as Axes3D

class GUI():
    def __init__(self, span):
        self.span = span
        self.fig = plt.figure()
        self.ax = Axes3D.Axes3D(self.fig)
        self.ax.set_xlim3d([-2.0, 2.0])
        self.ax.set_xlabel('X')
        self.ax.set_ylim3d([-2.0, 2.0])
        self.ax.set_ylabel('Y')
        self.ax.set_zlim3d([0, 5.0])
        self.ax.set_zlabel('Z')
        self.ax.set_title('Quadcopter Simulation')

        self.l1 = self.ax.plot([],[],[],color='blue',linewidth=3,antialiased=False)[0]
        self.l2 = self.ax.plot([],[],[],color='red',linewidth=3,antialiased=False)[0]
        self.hub = self.ax.plot([],[],[],marker='o',color='green', markersize=6,antialiased=False)[0]

    def update(self, state):
        R = state['rotation_matrix']
        span = self.span

        points = np.array([ [-span,0,0], [span,0,0], [0,-span,0], [0,span,0], [0,0,0], [0,0,0] ]).T
        points = np.dot(R,points)

        points[0,:] += state['position'][0]
        points[1,:] += state['position'][1]
        points[2,:] += state['position'][2]

        self.l1.set_data(points[0,0:2],points[1,0:2])
        self.l1.set_3d_properties(points[2,0:2])
        self.l2.set_data(points[0,2:4],points[1,2:4])
        self.l2.set_3d_properties(points[2,2:4])
        self.hub.set_data(points[0,5],points[1,5])
        self.hub.set_3d_properties(points[2,5])

        plt.pause(0.000000000000001)
