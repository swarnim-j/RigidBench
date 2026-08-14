import pymunk


def solve_layout(footprints, constraints, steps=80, dt=1 / 60):
    """Pymunk zero-gravity relaxation: nudge overlapping circle footprints apart."""
    space = pymunk.Space()
    space.gravity = (0, 0)
    space.damping = 0.1
    margin = 0.005

    bodies = {}
    for fp in footprints:
        if fp.get("static", False):
            body = pymunk.Body(body_type=pymunk.Body.STATIC)
        else:
            r = fp["radius"] + margin
            body = pymunk.Body(mass=1.0, moment=pymunk.moment_for_circle(1.0, 0, r))
        body.position = (fp["x"], fp["y"])
        shape = pymunk.Circle(body, fp["radius"] + margin)
        shape.elasticity = 0.0
        shape.friction = 1.0
        space.add(body, shape)
        bodies[fp["name"]] = body

    for c in constraints:
        a_body = bodies.get(c.get("a"))
        b_body = bodies.get(c.get("b"))
        if not a_body or not b_body:
            continue
        if c["type"] == "in_front_of":
            gap = c.get("min_gap", 0.05)
            space.add(pymunk.SlideJoint(b_body, a_body, (0, 0), (0, 0), gap, float("inf")))

    for fp in footprints:
        if fp.get("static", False):
            continue
        body = bodies[fp["name"]]
        space.add(
            pymunk.DampedSpring(
                space.static_body,
                body,
                (fp["x"], fp["y"]),
                (0, 0),
                rest_length=0,
                stiffness=50.0,
                damping=10.0,
            )
        )

    for _ in range(steps):
        space.step(dt)

    return {name: (body.position.x, body.position.y) for name, body in bodies.items()}
