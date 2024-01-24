from graphics import *
import numpy as np

from entities.boat import Boat
from entities.wind import Wind
from entities.world import World
from environment import SeabedMap
from tools.utils import *

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
        self.undraw = []
    
    def to_canvas(self, position):
        x = scale(self.world_width, self.win.width, position[0])
        y = scale(self.world_height, self.win.height, position[1])
        x = x + (0.5 * self.win.width)
        y = -y + (0.5 * self.win.height)
        return np.array([x, y])
    
    def clear(self):
        for item in self.undraw:
            item.undraw()
        self.win.update()
    
    def draw_polygon(self, vertices, pivot, angle, color, undraw=True):
        vertices = rotate_polygon(vertices, angle, pivot)
        vertices_canvas = [array_to_point(p) for p in map(lambda p: self.to_canvas(p), vertices)]
        draw = Polygon(vertices_canvas)
        if undraw:
            self.undraw.append(draw)
        draw.setFill(color)
        draw.setOutline(color)
        draw.draw(self.win)
    
    def draw_triangle(self, width: int, height: int, center, pivot, angle, color, undraw=True):
        top = np.array([center[0], center[1] + (height * 0.5)])
        lb = np.array([center[0] - width * 0.5, center[1] - (height * 0.5)])
        rb = np.array([center[0] + width * 0.5, center[1] - (height * 0.5)])
        vertices = [top, rb, lb]
        self.draw_polygon(vertices, pivot, angle, color, undraw)

    def draw_rectangle(self, width: int, height: int, center, pivot, angle, color, undraw=True):
        ul = np.array([center[0] - (width * 0.5), center[1] + (height * 0.5)])
        ur = np.array([center[0] + (width * 0.5), center[1] + (height * 0.5)])
        dr = np.array([center[0] + (width * 0.5), center[1] - (height * 0.5)])
        dl = np.array([center[0] - (width * 0.5), center[1] - (height * 0.5)])
        vertices = [ul, ur, dr, dl]
        self.draw_polygon(vertices, pivot, angle, color, undraw)

    def draw_map(self, map: SeabedMap):
        ul = np.array([map.min_x, map.min_y])
        ur = np.array([map.max_x, map.min_y])
        dr = np.array([map.max_x, map.max_y])
        dl = np.array([map.min_x, map.max_y])
        vertices = [ul, ur, dr, dl]
        self.draw_polygon(vertices, [0, 0], 0, color_rgb(119,255,243), undraw=False)
    
    def draw_wind(self, wind: Wind, center):
        width = 5
        height = 15
        color = color_rgb(15,15,15)
        angle = compute_angle(wind.velocity) - (np.pi * 0.5)
        self.draw_triangle(width, height, center, center, angle, color)
        
    def draw_boat(self, boat: Boat):
        # draw boat
        boat_width = boat.length
        boat_height = boat_width / 3
        boat_color = color_rgb(255,168,168)
        boat_angle = compute_angle(boat.heading)
        self.draw_rectangle(boat_width, boat_height, boat.position, boat.position, boat_angle, boat_color)

        # draw wing
        wing_height = boat_width / 2
        wing_width = wing_height / 3
        wing_color = color_rgb(150, 150, 150)
        wing_angle = boat.wing.controller.get_angle() + boat_angle
        self.draw_rectangle(wing_width, wing_height, boat.position, boat.position, wing_angle, wing_color)

        if self.debug:
            # draw velocity
            self.draw_vector(boat.position, boat.velocity, 'green', 2)
            # draw heading
            self.draw_vector(boat.position, boat.heading, 'red', 10)
            # draw rudder
            rudder_angle_rel = boat.rudder.controller.get_angle()
            rudder_angle_abs = -rudder_angle_rel + boat_angle
            rudder_angle_abs += np.pi
            rudder_heading = polar_to_cartesian(1, rudder_angle_abs) 
            self.draw_vector(boat.position, rudder_heading, 'purple', 10)
    
            # collision_box_center = self.to_canvas(boat.position)
            # radius = scale(self.world_width, self.win.width, boat.length * 0.5)
            # draw = Circle(Point(collision_box_center[0], collision_box_center[1]), radius)
            # draw.draw(self.win)
    
    def draw_vector(self, start, vec, color, gain=1, undraw=True):
        end = start + (vec * gain)
        start_canvas = self.to_canvas(start)
        end_canvas = self.to_canvas(end)
        draw = Line(array_to_point(start_canvas), array_to_point(end_canvas))
        if undraw:
            self.undraw.append(draw)
        draw.setFill(color)
        draw.setOutline(color)
        draw.draw(self.win)
    
    def draw_route(self, points, color, undraw=True):
        for i in range(len(points) - 1):
            start_canvas = self.to_canvas(points[i])
            end_canvas = self.to_canvas(points[i+1])
            draw = Line(array_to_point(start_canvas), array_to_point(end_canvas))
            if undraw:
                self.undraw.append(draw)
            draw.setFill(color)
            draw.setOutline(color)
            draw.draw(self.win)
    
    def draw_axis(self):
        x_axis = Line(Point(0, self.win.height * 0.5), Point(self.win.width, self.win.height * 0.5))
        y_axis = Line(Point(self.win.width * 0.5, 0), Point(self.win.width * 0.5, self.win.height))
        x_axis.draw(self.win)
        y_axis.draw(self.win)

        steps = 30

        x_step = self.win.width / steps
        x_position = 0
        offset_from_x_axis = 10
        for x in np.arange(0, self.win.width, x_step):
            value = scale(self.win.width, self.world_width, x) - (self.world_width * 0.5)
            p = Point(x, self.win.height * 0.5 + offset_from_x_axis)
            txt = Text(p, str(np.floor(value)))
            txt.setSize(8)
            txt.draw(self.win)
            x_position += x
        
        y_step = self.win.height / steps
        y_position = 0
        offset_from_y_axis = 10
        for y in np.arange(0, self.win.height, y_step):
            value = scale(self.win.height, self.world_height, y) - (self.world_height * 0.5)
            p = Point(self.win.width * 0.5 + offset_from_y_axis, y)
            txt = Text(p, str(np.floor(-value)))
            txt.setSize(8)
            txt.draw(self.win)
            y_position += y
    
    def draw_target(self, position):
        position_canvas = self.to_canvas(position)
        draw = Circle(Point(position_canvas[0], position_canvas[1]), 10)
        self.undraw.append(draw)
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
