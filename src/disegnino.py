from graphics import *
import numpy as np

from entities import compute_angle, Boat

class Drawer:
    def __init__(self, width: int, height: int):
        self.win = GraphWin("roadmap", width, height)
    
    def clear(self):
        for item in self.win.items:
            item.undraw()
        self.win.update()
    
    def draw_boat(self, boat: Boat):
        width = 30
        length = 60
        ul = Point(boat.position[0] - (width*0.5), boat.position[1] + (length*0.5))
        ur = Point(boat.position[0] + (width*0.5), boat.position[1] + (length*0.5))
        dr = Point(boat.position[0] + (width*0.5), boat.position[1] - (length*0.5))
        dl = Point(boat.position[0] - (width*0.5), boat.position[1] - (length*0.5))
        angle = compute_angle(boat.heading)
        vertices = rotate_polygon([ul, ur, dr, dl], angle)
        poly = Polygon(vertices)
        poly.setFill(color_rgb(255,168,168))
        poly.setOutline(color_rgb(255,168,168))
        poly.draw(self.win)

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
