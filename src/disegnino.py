from graphics import *
import numpy as np
import time

from boat import Boat

def draw_boat(win, boat: Boat):
    width = 30
    length = 60
    ul = Point(boat.position[0] - (width*0.5), boat.position[1] + (length*0.5))
    ur = Point(boat.position[0] + (width*0.5), boat.position[1] + (length*0.5))
    dr = Point(boat.position[0] + (width*0.5), boat.position[1] - (length*0.5))
    dl = Point(boat.position[0] - (width*0.5), boat.position[1] - (length*0.5))
    vertices = [ul, ur, dr, dl]
    poly = Polygon(vertices)
    poly.setFill(color_rgb(255,168,168))
    poly.setOutline(color_rgb(255,168,168))
    poly.draw(win)
    return poly

def rotate_polygon(vertices: list[Point], angle):
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


def main():

    width = 300
    height = 300

    win = GraphWin("roadmap", width, height)

    boat = Boat()

    poly = draw_boat(win, boat)

    dt = 1
    speed = np.array([10, 10])
    angle = 0.3
    while True:
        poly.undraw()
        poly.points = rotate_polygon(poly.points, angle * dt)
        poly.draw(win)
        s = speed * dt
        poly.move(s[0], s[1])
        time.sleep(dt)

main()