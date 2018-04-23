
#Documenting some of my basic tensorflow programs 

1. Modelling y=wx+b in tensor flow
#y_is_wx.py models a linear regression in tensor flow for the equation y=wx
datagen_y_is_wx.py generates data for this equation into foo.csv

To execute:
a. Run datagen_y_is_wx.py to generate foo.csv
b. Run y_is_wx.py

At the end of the run, you need to see cost nearly 0 and the optimized weights
for W and b would be output.
Also, a plot costs.png is generated which shows how cost would decrease with
each iteration.

2. Modelling y=w1x1+w2x2+b in tensor flow
#y_is_w1x_w2x2.py models a linear regression in tensor flow for the equation
y=w1x1+w2x2+b
datagen_y_is_w1x1_w2x2.py generates data for this equation into foo.csv

To execute:
a. Run datagen_y_is_w1x1_w2x2.py to generate foo.csv
b. Run y_is_w1x1_w2x2.py

At the end of the run, you need to see cost nearly 0 and the optimized weights
for W (a matrix containing W1 and W2) and b would be output.
Also, a plot costs.png is generated which shows how cost would decrease with
each iteration.


