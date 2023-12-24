from graphics import *
import numpy as np

from entities import Wind, Wing, compute_angle, Boat
from utils import cartesian_to_polar, polar_to_cartesian

def scale(from_, to_, value):
    return (to_/from_) * value

def array_to_point(point):
    return Point(point[0], point[1])

class Drawer:
    def __init__(self, canvas_width: int, canvas_height: int, world_width, world_height):
        self.win = GraphWin("roadmap", canvas_width, canvas_height)
        self.world_width = world_width
        self.world_height = world_height
        self.debug = False
    
    def to_canvas(self, position):
        x = scale(self.world_width, self.win.width, position[0])
        y = scale(self.world_height, self.win.height, position[1])
        x = x + (0.5 * self.win.width)
        y = -y + (0.5 * self.win.height)
        return np.array([x, y])
    
    def clear(self):
        for item in self.win.items:
            item.undraw()
        self.win.update()
    
    def draw_polygon(self, vertices, pivot, angle, color):
        vertices = rotate_polygon(vertices, angle, pivot)
        vertices_canvas = [array_to_point(p) for p in map(lambda p: self.to_canvas(p), vertices)]
        draw = Polygon(vertices_canvas)
        draw.setFill(color)
        draw.setOutline(color)
        draw.draw(self.win)
    
    def draw_triangle(self, width: int, height: int, center, pivot, angle, color):
        top = np.array([center[0], center[1] + (height * 0.5)])
        lb = np.array([center[0] - width * 0.5, center[1] - (height * 0.5)])
        rb = np.array([center[0] + width * 0.5, center[1] - (height * 0.5)])
        vertices = [top, rb, lb]
        print(center)
        self.draw_polygon(vertices, pivot, angle, color)

    def draw_rectangle(self, width: int, height: int, center, pivot, angle, color):
        ul = np.array([center[0] - (width * 0.5), center[1] + (height * 0.5)])
        ur = np.array([center[0] + (width * 0.5), center[1] + (height * 0.5)])
        dr = np.array([center[0] + (width * 0.5), center[1] - (height * 0.5)])
        dl = np.array([center[0] - (width * 0.5), center[1] - (height * 0.5)])
        vertices = [ul, ur, dr, dl]
        self.draw_polygon(vertices, pivot, angle, color)
    
    def draw_wind(self, wind: Wind, center):
        width = 2
        height = 30
        color = color_rgb(15,15,15)
        angle = compute_angle(wind.velocity) - (np.pi * 0.5)
        self.draw_triangle(width, height, center, center, angle, color)
        
    def draw_boat(self, boat: Boat):
        # draw boat
        boat_width = 15
        boat_height = 5
        boat_color = color_rgb(255,168,168)
        boat_angle = compute_angle(boat.heading)
        self.draw_rectangle(boat_width, boat_height, boat.position, boat.position, boat_angle, boat_color)

        wing_width = 2
        wing_height = 10
        wing_color = color_rgb(150, 150, 150)
        wing_angle = boat.wing.get_angle() + boat_angle
        self.draw_rectangle(wing_width, wing_height, boat.position, boat.position, wing_angle, wing_color)

        if self.debug:
            self.draw_vector(boat.position, boat.velocity, 'green', 2)
            # self.draw_vector(boat.position, boat.heading, 'purple', 10)
            rudder_angle_rel = compute_angle(boat.rudder.get_heading())
            rudder_angle_abs = rudder_angle_rel + boat_angle
            rudder_angle_abs += np.pi
            rudder_heading = polar_to_cartesian(1, rudder_angle_abs) 
            self.draw_vector(boat.position, rudder_heading, 'blue', 10)
    
    def draw_vector(self, start, vec, color, gain=1):
        end = start + (vec * gain)
        start_canvas = self.to_canvas(start)
        end_canvas = self.to_canvas(end)
        draw = Line(array_to_point(start_canvas), array_to_point(end_canvas))
        draw.setFill(color)
        draw.setOutline(color)
        draw.draw(self.win)
    
    def draw_axis(self):
        x_axis = Line(Point(0, self.win.height * 0.5), Point(self.win.width, self.win.height * 0.5))
        y_axis = Line(Point(self.win.width * 0.5, 0), Point(self.win.width * 0.5, self.win.height))
        x_axis.draw(self.win)
        y_axis.draw(self.win)

        steps = 10

        x_step = self.win.width / steps
        x_position = 0
        offset_from_x_axis = 10
        for x in np.arange(0, self.win.width, x_step):
            value = scale(self.win.width, self.world_width, x) - (self.world_width * 0.5)
            p = Point(x, self.win.height * 0.5 + offset_from_x_axis)
            txt = Text(p, str(value))
            txt.setSize(8)
            txt.draw(self.win)
            x_position += x
        
        y_step = self.win.height / steps
        y_position = 0
        offset_from_y_axis = 10
        for y in np.arange(0, self.win.height, y_step):
            value = scale(self.win.height, self.world_height, y) - (self.world_height * 0.5)
            p = Point(self.win.width * 0.5 + offset_from_y_axis, y)
            txt = Text(p, str(value))
            txt.setSize(8)
            txt.draw(self.win)
            y_position += y
    
    def draw_target(self, position):
        position_canvas = self.to_canvas(position)
        print(position_canvas)
        draw = Circle(Point(position_canvas[0], position_canvas[1]), 10)
        draw.draw(self.win)
    
        
def rotate_polygon(vertices, angle, pivot):
    # Create the rotation matrix
    cos_theta = np.cos(angle)
    sin_theta = np.sin(angle)

    # Rotate each vertex around the center
    rotated_vertices = [
        np.array([
            pivot[0] + (p[0] - pivot[0]) * cos_theta - (p[1] - pivot[1]) * sin_theta,
            pivot[1] + (p[0] - pivot[0]) * sin_theta + (p[1] - pivot[1]) * cos_theta
        ])
        for p in vertices
    ]

    return rotated_vertices
