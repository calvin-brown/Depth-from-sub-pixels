# Depth-from-sub-pixels

Obtaining depth information for a 2D image or set of images is a common task in computer vision. While many approaches exist for obtaining depth (perhaps most noteworthy is the growing prevalence of LIDAR as the cost comes down), one of the simplest is depth from stereo or triangulation. This is what the human visual system uses (in part) to determine how far away objects are.

By taking two images from two cameras (or eyes) with a known separation (called the baseline), the distance to an object can be determined from the shift in its location in the two images.
