# MIT License

# Copyright (c) 2020 Joseph Auckley, Matthew O'Kelly, Aman Sinha, Hongrui Zheng, Zirui Zang

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


"""
Generates random tracks.
Adapted from https://gym.openai.com/envs/CarRacing-v0
Author: Hongrui Zheng, Zirui Zang
Require shapely==1.7.1
"""

import cv2
import os
import math
import numpy as np
import shapely.geometry as shp
import matplotlib.pyplot as plt
import argparse
from frenet_utils import *


LOAD_F1TENTH_SCALE = False
PLOT_FOR_SHOW = True

WIDTH = 5.0 # half width
# CHECKPOINTS = np.random.randint(5, 20)
CHECKPOINTS = 25
ROUNDNESS = 0.1
TRACK_RADIUS = np.maximum(CHECKPOINTS*4, 40)
TRACK_DETAIL_STEP = 5
TRACK_TURN_RATE = np.random.uniform(0.05, 0.95)
MAP_IMG_RESOLUTION = 0.5 # for f1tenth car, also use 0.5. Only change scale when reading the map.

GENERATE_OBSTACLES = True
OBSTACLE_INTERVAL = 30 # average distance between obstacles
LEAST_GAP_WIDTH = 4 # least gap width between obstacles and boundary
    
class TrackGen:
    def __init__(self):
        self.ind = 0
        if not os.path.exists('gen_maps'):
            print('Creating gen_maps/ directory.')
            os.makedirs('gen_maps')
        
    def load_map_random_gen(self, MAP_DIR, map_name, f1tenth_scale=False, load_obs=False):
        import yaml, cv2
        with open(MAP_DIR + map_name + '.yaml') as stream:
            info = yaml.load(stream, Loader=yaml.Loader)

        if f1tenth_scale:
            scale = 1/7
        else:
            scale = 1
        cv_img = cv2.imread(MAP_DIR + info['image'], -1)
        if load_obs:
            obs_list = np.loadtxt(MAP_DIR + map_name + '_obs.csv', delimiter=',', skiprows=0) * scale
        else:
            obs_list = []
        waypoints = np.loadtxt(MAP_DIR + map_name + '.csv', delimiter=',', skiprows=0) * scale
        resolution = info['resolution'] * scale
        map_origin = np.array(info['origin_pix']) * resolution
    
        return cv_img, waypoints, obs_list, map_origin, resolution
        
    def create_track(self):
        start_alpha = 0.
        # Create checkpoints
        checkpoints = []
        for c in range(CHECKPOINTS):
            alpha = 2*math.pi*c/CHECKPOINTS + np.random.uniform(0, 2*math.pi*1/CHECKPOINTS)
            rad = np.random.uniform(TRACK_RADIUS/2, TRACK_RADIUS)
            if c==0:
                alpha = 0
                rad = np.random.uniform(1*TRACK_RADIUS, 1.5*TRACK_RADIUS)
            if c==CHECKPOINTS-1:
                alpha = 2*math.pi*c/CHECKPOINTS
                start_alpha = 2*math.pi*(-0.5)/CHECKPOINTS
                rad = np.random.uniform(1*TRACK_RADIUS, 1.5*TRACK_RADIUS)
            checkpoints.append( (alpha, rad*math.cos(alpha), rad*math.sin(alpha)) )
        road = []

        # Go from one checkpoint to another to create track
        x, y, beta = 1.5*TRACK_RADIUS, 0, 0
        dest_i = 0
        laps = 0
        track = []
        no_freeze = 2500
        visited_other_side = False
        while True:
            alpha = math.atan2(y, x)
            if visited_other_side and alpha > 0:
                laps += 1
                visited_other_side = False
            if alpha < 0:
                visited_other_side = True
                alpha += 2*math.pi
            while True:
                failed = True
                while True:
                    dest_alpha, dest_x, dest_y = checkpoints[dest_i % len(checkpoints)]
                    if alpha <= dest_alpha:
                        failed = False
                        break
                    dest_i += 1
                    if dest_i % len(checkpoints) == 0:
                        break
                if not failed:
                    break
                alpha -= 2*math.pi
                continue
            r1x = math.cos(beta)
            r1y = math.sin(beta)
            p1x = -r1y
            p1y = r1x
            dest_dx = dest_x - x
            dest_dy = dest_y - y
            proj = r1x*dest_dx + r1y*dest_dy
            while beta - alpha >  1.5*math.pi:
                beta -= 2*math.pi
            while beta - alpha < -1.5*math.pi:
                beta += 2*math.pi
            prev_beta = beta
            proj /= ROUNDNESS
            if proj >  0.3:
                beta -= min(TRACK_TURN_RATE, abs(0.001*proj))
            if proj < -0.3:
                beta += min(TRACK_TURN_RATE, abs(0.001*proj))
            x += p1x*TRACK_DETAIL_STEP
            y += p1y*TRACK_DETAIL_STEP
            track.append( (alpha,prev_beta*0.5 + beta*0.5,x,y) )
            if laps > 4:
                break
            no_freeze -= 1
            if no_freeze==0:
                break

        # Find closed loop
        i1, i2 = -1, -1
        i = len(track)
        while True:
            i -= 1
            if i==0:
                return False
            pass_through_start = track[i][0] > start_alpha and track[i-1][0] <= start_alpha
            if pass_through_start and i2==-1:
                i2 = i
            elif pass_through_start and i1==-1:
                i1 = i
                break
        print("Track generation: %i..%i -> %i-tiles track" % (i1, i2, i2-i1))
        assert i1!=-1
        assert i2!=-1

        track = track[i1:i2-1]
        first_beta = track[0][1]
        first_perp_x = math.cos(first_beta)
        first_perp_y = math.sin(first_beta)

        # Length of perpendicular jump to put together head and tail
        well_glued_together = np.sqrt(
            np.square( first_perp_x*(track[0][2] - track[-1][2]) ) +
            np.square( first_perp_y*(track[0][3] - track[-1][3]) ))
        if well_glued_together > TRACK_DETAIL_STEP:
            return False

        # post processing, converting to numpy, finding exterior and interior walls
        track_xy = [(x, y) for (a1, b1, x, y) in track]
        track_xy = np.asarray(track_xy)
        track_poly = shp.Polygon(track_xy)
        track_xy_offset_in = track_poly.buffer(WIDTH)
        # track_xy_offset_zero = track_poly.buffer(0)
        track_xy_offset_out = track_poly.buffer(-WIDTH)
        # track_xy = np.array(track_xy_offset_zero.exterior)
        track_xy_offset_in_np = np.array(track_xy_offset_in.exterior)
        track_xy_offset_out_np = np.array(track_xy_offset_out.exterior)
        return track_xy, track_xy_offset_in_np, track_xy_offset_out_np
    
    def convert_track(self, track, track_int, track_ext, iter):
        resolution = MAP_IMG_RESOLUTION
        plot_gen_dpi = 10 * resolution
        xy_limit = 200
        image_size_x = 2*xy_limit/resolution/plot_gen_dpi

        print('track', track.shape)
        # converts track to image and saves the centerline as waypoints
        fig, ax = plt.subplots(dpi=plot_gen_dpi)
        fig.set_size_inches(image_size_x, image_size_x)
        ax.plot(*track_int.T, color='black', linewidth=200/plot_gen_dpi)
        ax.plot(*track_ext.T, color='black', linewidth=200/plot_gen_dpi)
        plt.tight_layout()
        ax.set_aspect('equal')
        ax.set_xlim(-xy_limit, xy_limit)
        ax.set_ylim(-xy_limit, xy_limit)
        plt.axis('off')
        
        # map_width, map_height = fig.canvas.get_width_height()
        # print('map size: ', map_width, map_height)

        # transform the track center line into pixel coordinates
        xy_pixels = ax.transData.transform(track)
        origin_x_pix = xy_pixels[0, 0]
        origin_y_pix = xy_pixels[0, 1]
        xy_pixels = xy_pixels - np.array([[origin_x_pix, origin_y_pix]])
        last_point = (xy_pixels[0] + xy_pixels[-1])/2
        xy_pixels = np.vstack([xy_pixels, last_point[None, :]])
        xy_pixels = np.vstack([xy_pixels, xy_pixels[0]])
        
        # increase centerline density
        for _ in range(1):
            new_xy_pixels = []
            for ind in range(len(xy_pixels)-1):
                new_xy_pixels.append(xy_pixels[ind])
                new_xy_pixels.append((xy_pixels[ind] + xy_pixels[ind+1])/2)
            xy_pixels = np.array(new_xy_pixels)
            
        map_origin_x = -origin_x_pix*resolution
        map_origin_y = -origin_y_pix*resolution
        
        plt.savefig('gen_maps/map' + str(iter) + '.png', dpi=plot_gen_dpi)

        # # convert image using cv2
        cv_img = cv2.imread('gen_maps/map' + str(iter) + '.png', -1)
        # # convert to bw
        cv_img_bw = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
        # # saving to img
        cv2.imwrite('gen_maps/map' + str(iter) + '.png', cv_img_bw)

        # create yaml file
        yaml = open('gen_maps/map' + str(iter) + '.yaml', 'w')
        yaml.write('image: map' + str(iter) + '.png\n')
        yaml.write(f'resolution: {resolution} \n')
        yaml.write('origin: [' + str(map_origin_x) + ',' + str(map_origin_y) + ', 0.000000]\n')
        yaml.write('origin_pix: [' + str(-origin_x_pix) + ',' + str(-origin_y_pix) + ', 0.000000]\n')
        yaml.write('negate: 0\noccupied_thresh: 0.45\nfree_thresh: 0.196')
        yaml.close()
        
        if GENERATE_OBSTACLES:
            yaml = open('gen_maps/map_obs' + str(iter) + '.yaml', 'w')
            yaml.write('image: map_obs' + str(iter) + '.png\n')
            yaml.write(f'resolution: {resolution} \n')
            yaml.write('origin: [' + str(map_origin_x) + ',' + str(map_origin_y) + ', 0.000000]\n')
            yaml.write('origin_pix: [' + str(-origin_x_pix) + ',' + str(-origin_y_pix) + ', 0.000000]\n')
            yaml.write('negate: 0\noccupied_thresh: 0.45\nfree_thresh: 0.196')
            yaml.close()
        
        # saving track centerline as a csv in ros coords
        waypoints_csv = open('gen_maps/map' + str(iter) + '.csv', 'w')
        for row in xy_pixels:
            waypoints_csv.write(str(resolution*row[0]) + ', ' + str(resolution*row[1]) + ', ' + str(WIDTH) + ', ' + str(WIDTH) + '\n')
        waypoints_csv.close()
        
        if GENERATE_OBSTACLES:
            waypoints_csv = open('gen_maps/map_obs' + str(iter) + '.csv', 'w')
            for row in xy_pixels:
                waypoints_csv.write(str(resolution*row[0]) + ', ' + str(resolution*row[1]) + ', ' + str(WIDTH) + ', ' + str(WIDTH) + '\n')
            waypoints_csv.close()
            
            # get frenet frame from centerline to generate obstacles
            centerline_traj = []
            for ind in range(track.shape[0]):
                centerline_traj.append([track[ind, 0], track[ind, 1], WIDTH, WIDTH])    
            centerline_traj.append([(track[0, 0] + track[-1, 0])/2, (track[0, 1]+track[-1, 1])/2, WIDTH, WIDTH])  
            centerline_traj.append([track[0, 0], track[0, 1], WIDTH, WIDTH])    
            centerline_traj = np.array(centerline_traj)
            frenet_traj = centerline_to_frenet(centerline_traj)
            obsta_num = int(frenet_traj[-1, 0] / OBSTACLE_INTERVAL)
            obsta_list = []
            obsta_xy = frenet_to_cartesian([0, 0, 0], frenet_traj)
            obsta_list.append([obsta_xy[0], obsta_xy[1], LEAST_GAP_WIDTH]) # add a dummy obstacle at start
            
            while len(obsta_list) < obsta_num+1:
                obs2obs_check = 1
                while obs2obs_check == 1:
                    obsta_s = np.random.uniform(0, frenet_traj[-1, 0])
                    obsta_ey = np.random.uniform(-WIDTH, WIDTH)
                    radius_max = np.abs(obsta_ey) - LEAST_GAP_WIDTH + WIDTH
                    obsta_radius = np.random.uniform(np.minimum(2, radius_max), radius_max)
                    obsta_xy = frenet_to_cartesian([obsta_s, obsta_ey, 0], frenet_traj)
                    
                    # check 
                    obs2obs_check = 0
                    for obsta in obsta_list:
                        if np.linalg.norm(obsta[:2] - obsta_xy[:2]) - obsta[2] - obsta_radius < LEAST_GAP_WIDTH:
                            obs2obs_check = 1
                if obsta_radius > 0:
                    obsta_list.append([obsta_xy[0], obsta_xy[1], obsta_radius])

            for obsta in obsta_list[1:]:
                patch = plt.Circle((obsta[0], obsta[1]), obsta[2], color='k')
                ax.add_patch(patch)
            
            plt.savefig('gen_maps/map_obs' + str(iter) + '.png', dpi=plot_gen_dpi)
            cv_img = cv2.imread('gen_maps/map_obs' + str(iter) + '.png', -1)
            cv_img_bw = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
            cv2.imwrite('gen_maps/map_obs' + str(iter) + '.png', cv_img_bw)
            
            ax.plot(centerline_traj[:, 0], centerline_traj[:, 1], 'r.', markersize=200/plot_gen_dpi)
            
            
            # save obstacle locations and sizes
            obstacles_csv = open('gen_maps/map_obs' + str(iter) + '_obs.csv', 'w')
            obsta_list = np.array(obsta_list)
            xy_pixels_obs = ax.transData.transform(obsta_list[1:, :2])
            for ind, row in enumerate(xy_pixels_obs):
                obstacles_csv.write(str(row[0]*resolution+map_origin_x) + ', ' + str(row[1]*resolution+map_origin_y) + ', ' + str(obsta_list[ind+1, 2]) + '\n')
            obstacles_csv.close()
        
        if PLOT_FOR_SHOW: plt.show()
        plt.close()
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0, help='Seed for the numpy rng.')
    parser.add_argument('--num_maps', type=int, default=1, help='Number of gen_maps to generate.')
    args = parser.parse_args()
    if args.seed != 0:
        np.random.seed(args.seed)
    track_gen = TrackGen()
    
    ind = 0
    while ind < args.num_maps:
        try:
            track, track_int, track_ext = track_gen.create_track()
        except:
            print('Random generator failed, retrying')
            continue
        track_gen.convert_track(track, track_int, track_ext, ind)
        
        if PLOT_FOR_SHOW: 
            # Example use case
            # Conversion from map coordinate to image pixel coordinate and then plot it in a image format (row col instead of x y)
            # [ ((x_map - map_origin_x) / resolution), (-(y_map - map_origin_y) / resolution) - image_size_y ]
            if GENERATE_OBSTACLES:
                filename = 'map_obs'
            else:
                filename = 'map'
            cv_img_obs, waypoints, obs_list, map_origin, resolution = track_gen.load_map_random_gen('gen_maps/', filename + str(ind), f1tenth_scale=LOAD_F1TENTH_SCALE, load_obs=GENERATE_OBSTACLES)
            fig, ax2 = plt.subplots(dpi=100)
            ax2.imshow(cv_img_obs)
            ax2.plot(((waypoints[:, 0] - map_origin[0])) / resolution , (-(waypoints[:, 1] - map_origin[1]) / resolution) + cv_img_obs.shape[1] , 'r.', markersize=1)    
            if GENERATE_OBSTACLES:
                for obsta in obs_list:
                    patch = plt.Circle(((obsta[0] - map_origin[0]) / resolution, -(obsta[1]- map_origin[1])/ resolution+ cv_img_obs.shape[1]) , obsta[2]/resolution, color='r', fill=False)
                    ax2.add_patch(patch)
            patch = plt.Circle((-map_origin[0] / resolution, map_origin[1]/ resolution+ cv_img_obs.shape[1]) , 1/resolution, color='g', fill=True)
            ax2.add_patch(patch)
            if LOAD_F1TENTH_SCALE:
                str1 = ' F1Tenth scale'
            else:
                str1 = ' 1:1 scale'
            ax2.set_title(filename + " " + str(ind) + str1)
            plt.tight_layout()
            # plt.savefig('gen_maps/example' + str(iter) + '.png', dpi=100)
            
            plt.show()
            plt.close()
            LOAD_F1TENTH_SCALE = True
            if GENERATE_OBSTACLES:
                filename = 'map_obs'
            else:
                filename = 'map'
            cv_img_obs, waypoints, obs_list, map_origin, resolution = track_gen.load_map_random_gen('gen_maps/', filename + str(ind), f1tenth_scale=LOAD_F1TENTH_SCALE, load_obs=GENERATE_OBSTACLES)
            fig, ax2 = plt.subplots(dpi=100)
            ax2.imshow(cv_img_obs)
            ax2.plot(((waypoints[:, 0] - map_origin[0])) / resolution , (-(waypoints[:, 1] - map_origin[1]) / resolution) + cv_img_obs.shape[1] , 'r.', markersize=1)    
            if GENERATE_OBSTACLES:
                for obsta in obs_list:
                    patch = plt.Circle(((obsta[0] - map_origin[0]) / resolution, -(obsta[1]- map_origin[1])/ resolution+ cv_img_obs.shape[1]) , obsta[2]/resolution, color='r', fill=False)
                    ax2.add_patch(patch)
            patch = plt.Circle((-map_origin[0] / resolution, map_origin[1]/ resolution+ cv_img_obs.shape[1]) , 1/resolution, color='g', fill=True)
            ax2.add_patch(patch)
            if LOAD_F1TENTH_SCALE:
                str1 = ' F1Tenth scale'
            else:
                str1 = ' 1:1 scale'
            ax2.set_title(filename + " " + str(ind) + str1)
            plt.tight_layout()
            # plt.savefig('gen_maps/example' + str(iter) + '.png', dpi=100)
            
            plt.show()
            plt.close()
        ind += 1






    
    
    
    
    

    