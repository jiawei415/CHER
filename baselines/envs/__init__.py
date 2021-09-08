from gym.envs.registration import register

def register_envs():
    register(
        id='SawyerDoorFixEnv-v1',
        entry_point='mher.envs.sawyer_door:SawyerDoorFixEnv',
    )
    register(
        id='SawyerReachXYEnv-v1',
        entry_point='mher.envs.sawyer_reach:SawyerReachXYEnv',
        tags={
            'git-commit-hash': '2d95c75',
            'author': 'murtaza'
        },
        kwargs={
            'hide_goal_markers': True,
            'norm_order': 2,
        },
    )
    register(
        id='PointMassEmptyEnv-v1',
        entry_point='mher.envs.pointmass:PointMassEnv',
        kwargs={
            'room_type': 'empty', # ['empty', 'wall', 'rooms']
        }
    )
    register(
        id='PointMassWallEnv-v1',
        entry_point='mher.envs.pointmass:PointMassEnv',
        kwargs={
            'room_type': 'wall',
        }
    )
    register(
        id='PointMassRoomsEnv-v1',
        entry_point='mher.envs.pointmass:PointMassEnv',
        kwargs={
            'room_type': 'rooms',
        }
    )
    register(
        id='Point2DLargeEnv-v1',
        entry_point='mher.envs.point2d:Point2DEnv',
        tags={
            'git-commit-hash': '4efe2be',
            'author': 'Vitchyr'
        },
        kwargs={
            'images_are_rgb': True,
            'target_radius': 1,
            'ball_radius': 0.5,
            'render_onscreen': False,
            'show_goal': True,
            'render_size':512,
            'get_image_base_render_size': (48, 48),
            'bg_color': 'white',
        },
    )
    register(
        id='Point2D-FourRoom-v1',
        entry_point='mher.envs.point2d:Point2DWallEnv',
        kwargs={
            'action_scale': 1,
            'wall_shape': 'four-room-v1', 
            'wall_thickness': 0.30,
            'target_radius':1,
            'ball_radius':0.5,
            'render_size': 512,
            'wall_color': 'darkgray',
            'bg_color': 'white',
            'images_are_rgb': True,
            'render_onscreen': False,
            'show_goal': True,
            'get_image_base_render_size': (48, 48),
            'boundary_dist':4,
        },
    )
