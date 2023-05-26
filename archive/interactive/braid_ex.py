from braid import *

data = 0, 0, 1, 0.8, 1, 0.2, 0, 0.4, 0.8, 1  # arbitrary timseries
f = timeseries(data)
plot(f)

t = Thread(1)
t.chord = D, SUSb9
t.pattern = [1, 2, 3, 4] * 4
t.start()

t.velocity = 0.0  # sets the lower bound of the range to 0.0
t.velocity = tween(
    1.0, 24, f
)  # sets the uppper bound of the range to 1.0, and applies the signal shape over 24 cycles


t = Thread(10)
t.start()


def neweuc(x, y, z):
    return [60 * i for i in euc(x, y, z)]


t.pattern = neweuc(8, 5, 43)
t.pattern = tween(neweuc(8, 6, 50), 8)
t.pattern = tween(neweuc(8, 5, 43), 8)


def ease_out(pos):
    pos = clamp(pos)
    return (pos - 1) ** 3 + 1


plot(ease_out)
