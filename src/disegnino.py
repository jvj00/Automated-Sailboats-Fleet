from graphics import *
import numpy as np

from entities import Wind, Wing, compute_angle, Boat
from utils import cartesian_to_polar, polar_to_cartesian

class Drawer:
    def __init__(self, width: int, height: int):
        self.win = GraphWin("roadmap", width, height)
        self.debug = False
    
    def clear(self):
        for item in self.win.items:
            item.undraw()
        self.win.update()
    
    def draw_polygon(self, vertices, pivot, angle, color):
        vertices = rotate_polygon(vertices, angle, pivot)
        draw = Polygon(vertices)
        draw.setFill(color)
        draw.setOutline(color)
        draw.draw(self.win)
    
    def draw_triangle(self, width: int, height: int, center, pivot, angle, color):
        top = Point(center[0], center[1] + (height * 0.5))
        lb = Point(center[0] - width * 0.5, center[1] - (height * 0.5))
        rb = Point(center[0] + width * 0.5, center[1] - (height * 0.5))
        vertices = [top, rb, lb]
        self.draw_polygon(vertices, pivot, angle, color)

    def draw_rectangle(self, width: int, height: int, center, pivot, angle, color):
        ul = Point(center[0] - (width * 0.5), center[1] + (height * 0.5))
        ur = Point(center[0] + (width * 0.5), center[1] + (height * 0.5))
        dr = Point(center[0] + (width * 0.5), center[1] - (height * 0.5))
        dl = Point(center[0] - (width * 0.5), center[1] - (height * 0.5))
        vertices = [ul, ur, dr, dl]
        self.draw_polygon(vertices, pivot, angle, color)
    
    def draw_wind(self, wind: Wind, center):
        width = 15
        height = 30
        color = color_rgb(15,15,15)
        angle = compute_angle(wind.velocity) - (np.pi * 0.5)
        self.draw_triangle(width, height, center, center, angle, color)
        
    def draw_boat(self, boat: Boat):
        # draw boat
        boat_width = 60
        boat_height = 30
        boat_color = color_rgb(255,168,168)
        boat_angle = compute_angle(boat.heading)
        self.draw_rectangle(boat_width, boat_height, boat.position, boat.position, boat_angle, boat_color)

        wing_width = 5
        wing_height = 25
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
        draw = Line(Point(start[0], start[1]), Point(end[0], end[1]))
        draw.setFill(color)
        draw.setOutline(color)
        draw.draw(self.win)

def rotate_polygon(vertices: list[Point], angle: float, pivot):
    # Create the rotation matrix
    cos_theta = np.cos(angle)
    sin_theta = np.sin(angle)

    # Rotate each vertex around the center
    rotated_vertices = [
        Point(
            pivot[0] + (p.x - pivot[0]) * cos_theta - (p.y - pivot[1]) * sin_theta,
            pivot[1] + (p.x - pivot[0]) * sin_theta + (p.y - pivot[1]) * cos_theta
        )
        for p in vertices
    ]

    return rotated_vertices
