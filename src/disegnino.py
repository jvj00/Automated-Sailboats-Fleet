from graphics import *
import numpy as np

from entities import Wind, Wing, compute_angle, Boat

class Drawer:
    def __init__(self, width: int, height: int):
        self.win = GraphWin("roadmap", width, height)
    
    def clear(self):
        for item in self.win.items:
            item.undraw()
        self.win.update()
    
    def draw_wind(self, wind: Wind, position):
        width = 15
        height = 30
        color = color_rgb(15,15,15)
        top = Point(position[0], position[1] + (height * 0.5))
        lb = Point(position[0] - width * 0.5, position[1] - (height * 0.5))
        rb = Point(position[0] + width * 0.5, position[1] - (height * 0.5))
        angle = compute_angle(wind.velocity) - (np.pi * 0.5)
        vertices = rotate_polygon([top, lb, rb], angle)
        draw = Polygon(vertices)
        draw.setFill(color)
        draw.setOutline(color)
        draw.draw(self.win)

    def draw_rectangle(self, width: int, height: int, position, heading, color):
        ul = Point(position[0] - (width * 0.5), position[1] + (height * 0.5))
        ur = Point(position[0] + (width * 0.5), position[1] + (height * 0.5))
        dr = Point(position[0] + (width * 0.5), position[1] - (height * 0.5))
        dl = Point(position[0] - (width * 0.5), position[1] - (height * 0.5))
        angle = compute_angle(heading) - (np.pi * 0.5)
        vertices = rotate_polygon([ul, ur, dr, dl], angle)
        draw = Polygon(vertices)
        draw.setFill(color)
        draw.setOutline(color)
        draw.draw(self.win)
        
    def draw_boat(self, boat: Boat):
        # draw boat
        width = 30
        height = 60
        color = color_rgb(255,168,168)
        self.draw_rectangle(width, height, boat.position, boat.heading, color)

        width = 25
        height = 5
        color = color_rgb(150, 150, 150)
        self.draw_rectangle(width, height, boat.position, boat.wing.get_heading(), color)

def rotate_polygon(vertices: list[Point], angle: float):
    # Calculate the center of the polygon
    cx = sum(p.x for p in vertices) / len(vertices)
    cy = sum(p.y for p in vertices) / len(vertices)

    # Create the rotation matrix
    cos_theta = np.cos(angle)
    sin_theta = np.sin(angle)

    # Rotate each vertex around the center
    rotated_vertices = [
        Point(
            cx + (p.x - cx) * cos_theta - (p.y - cy) * sin_theta,
            cy + (p.x - cx) * sin_theta + (p.y - cy) * cos_theta
        )
        for p in vertices
    ]

    return rotated_vertices
