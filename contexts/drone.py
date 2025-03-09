
def yaw_to_idx(yaw):
	while yaw < 0:
		yaw += 2*np.pi
	while yaw >= 2*np.pi:
		yaw -= 2*np.pi
	yaw_idx = round(yaw/(np.pi/2))
	return yaw_idx
