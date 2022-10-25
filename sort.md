# Using SORT in your own project

```
from sort import *

#create instance of SORT
mot_tracker = Sort() 

# get detections
...

# update SORT
track_bbs_ids = mot_tracker.update(detections)

# track_bbs_ids is a np array where each row contains a valid bounding box and track_id (last column)
...
```

SORT takes a bounding box in the centre form `[x,y,s,r]` and returns it in the form `[x1,y1,x2,y2]` where `x1,y1` is the top left and `x2,y2` is the bottom right.