import json
from bokeh.plotting import figure, show
from bokeh.layouts import row 

files = ["first_version.json","second_version.json","third_version.json","fourth_version.json","fifth_version.json","sixth_version.json","seventh_version.json"]
figures=[]
for file in files:
    with open(file,'r') as f:
        data=json.loads(f.read())
        f.close()
    print(data)

    white_balls_count={x:0 for x in range(1,70)}
    red_ball_count={x:0 for x in range(1,43)}

    for pick in data.values():
        white,red=pick
        for ball in white:
            white_balls_count[int(ball)]+=1
        red_ball_count[int(red)]+=1
    print("done")
    x,y=[],[]
    for k,v in white_balls_count.items():
        x.append(k)
        y.append(v)
    p=figure(title=f'{file} White Ball frequency', x_axis_label="Number", y_axis_label="Frequency")
    p.vbar(x=x,top=y,bottom=1,width=0.1)
    figures.append(p)
    x,y=[],[]
    for k,v in red_ball_count.items():
        x.append(k)
        y.append(v)
    p=figure(title=f'{file} Red Ball frequency', x_axis_label="Number", y_axis_label="Frequency")
    p.vbar(x=x,top=y,bottom=1,width=0.1)
    figures.append(p)

show(row(*figures))