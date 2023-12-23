from graphics import *
import numpy as np

from entities import Wind, Wing, compute_angle, Boat
from utils import cartesian_to_polar, polar_to_cartesian

class Drawer:
    def __init__(self, width: int, height: int):
        self.win = GraphWin("roadmap", width, height)
    
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
        width = 60
        height = 30
        color = color_rgb(255,168,168)
        boat_angle = compute_angle(boat.heading)
        self.draw_rectangle(width, height, boat.position, boat.position, boat_angle, color)

        width = 5
        height = 25
        color = color_rgb(150, 150, 150)
        wing_angle_rel = compute_angle(boat.wing.get_heading())
        wing_angle_abs = wing_angle_rel + boat_angle
        self.draw_rectangle(width, height, boat.position, boat.position, wing_angle_abs, color)
    
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
