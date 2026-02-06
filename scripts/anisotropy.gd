extends Node

# Anisotropy

func describe(image: Image) -> float:
	
	var h = image.get_height()
	var w = image.get_width()
	
	var xx := 0.0
	var yy := 0.0
	var xy := 0.0
	
	for y in range(h-1):
		for x in range(w-1):
			var dy := image.get_pixel(x, y+1).r - image.get_pixel(x, y).r
			var dx := image.get_pixel(x+1, y).r - image.get_pixel(x, y).r
			
			xx += dx * dx
			yy += dy * dy
			xy += dx * dy
	
	var mxx = xx / ( (h-1) * (w-1) )
	var myy = yy / ( (h-1) * (w-1) )
	var mxy = xy / ( (h-1) * (w-1) )
	
	var det = (myy+mxx)*(myy+mxx) - 4*(myy*mxx-mxy*mxy)
	var l1 = (myy+mxx + sqrt(det)) / 2
	var l2 = (myy+mxx - sqrt(det)) / 2
	
	return (l1 - l2) / (l1 + l2)
