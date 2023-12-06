import numpy as np
np.set_printoptions(precision=3,suppress=True)

class EKF:    
    def __init__(self, dt, initial_pos, initial_speed=np.array([0,0])):
        """Initialises the EFK for our robot model
        Args: 
            dt: float - time interval every time we compute a new measure
            initial_pos: np.array - initial position, with [pos_x, pos_y, orientation]
            initial_speed: np.array - initial speed, with [linear_speed, angular_speed]. Default to [0,0]
            """
        # A matrix
        # 3x3 matrix -> number of states x number of states matrix
        # Expresses how the state of the system [x,y,yaw] changes 
        # from k-1 to k when no control command is executed.
        self.A_k_minus_1 = np.array([[1.0,  0,   0],
                                    [  0,1.0,   0],
                                    [  0,  0, 1.0]])
    
        # Noise applied to the forward kinematics (calculation
        # of the estimated state at time k from the state
        # transition model of the mobile robot). This is a vector
        # with the number of elements equal to the number of states
        self.process_noise_v_k_minus_1 = np.array([0.01,0.01,0.003])     #bruit qu'on peut calibrer (model)

        # State model noise covariance matrix Q_k
        # When Q is large, the Kalman Filter tracks large changes in 
        # the sensor measurements more closely than for smaller Q.
        # Q is a square matrix that has the same number of rows as states.
        self.Q_k = np.array([[1.0,   0,   0],                           #bruit a calibrer (model)
                        [  0, 1.0,   0],
                        [  0,   0, 1.0]])
 
        # Measurement matrix H_k
        self.H_k = np.array([[1.0,  0,   0],
                        [  0,1.0,   0],
                        [  0,  0, 1.0]])
                                
        # Sensor measurement noise covariance matrix R_k
        # Has the same number of rows and columns as sensor measurements.
        # If we are sure about the measurements, R will be near zero.
        self.R_k = np.array([[1.0,   0,    0],                          # bruit a calibrer (camera)
                        [  0, 1.0,    0],
                        [  0,    0, 1.0]])
        
        # same matrix as just above, but for when the camera gives no input ( i.e. input 0,0,0) to trust more the model than the cam
        self.R_no_cam = np.array([[5.0, 0, 0],                          # bruit a calibrer (camera)
                                  [0, 5.0, 0],
                                  [0, 0, 5.0]])
                                
        # Sensor noise. 
        self.sensor_noise_w_k = np.array([0.07,0.07,0.04])              # bruit a calibrer (camera)
                # # We start at time k=1
                # k = 0
     
        # Time interval in seconds
        self.dk = dt
                     
        # The estimated state vector at time k-1 in the global reference frame.
        # starts with initial pos
        # [x_k_minus_1, y_k_minus_1, yaw_k_minus_1]
        # [pixels, pixels, degrees]
        self.state_estimate_k_minus_1 = initial_pos
     
        # The control input vector at time k-1 in the global reference frame.
        # [v, yaw_rate]
        # [pixels/second, deg/second]
        self.control_vector_k_minus_1 = initial_speed
     
        # State covariance matrix P_k_minus_1
        # It represents an estimate of 
        # the accuracy of the state estimate at time k made using the
        # state transition matrix. We start off with guessed values.
        self.P_k_minus_1 = np.array([[0.1,  0,   0],
                            [  0,0.1,   0],
                            [  0,  0, 0.1]])

    def getB(self, yaw, deltak):
        """
        Calculates and returns the B matrix
        3x2 matix -> number of states x number of control inputs
        The control inputs are the forward speed and the
        rotation rate around the z axis from the x-axis in the 
        counterclockwise direction.
        [v,yaw_rate]
        Expresses how the state of the system [x,y,yaw] changes
        from k-1 to k due to the control commands (i.e. control input).
        :param yaw: The yaw angle (rotation angle around the z axis) in deg
        :param deltak: The change in time from time step k-1 to k in sec
        """
        B = np.array([[np.cos(yaw)*deltak, 0],
                    [np.sin(yaw)*deltak, 0],
                    [0, deltak]])
        return B
    
    def ekf(self, z_k_observation_vector, control_vector_k_minus_1):
        """
        Extended Kalman Filter. Fuses noisy sensor measurement to 
        create an optimal estimate of the state of the robotic system.
            
        INPUT
            :param z_k_observation_vector The observation from the Odometry
                3x1 NumPy Array [x,y,yaw] in the global reference frame
                in [pixels,pixels,degrees].
            :param control_vector_k_minus_1 The control vector applied at time k-1
                3x1 NumPy Array [v,v,yaw rate] in the global reference frame
                in [pixels per second,pixels per second,degrees per second].
                
        OUTPUT
            :return state_estimate_k near-optimal state estimate at time k  
                3x1 NumPy Array ---> [pixels,pixels,degrees]
            :return P_k state covariance_estimate for time k
                3x3 NumPy Array                 
        """
        ######################### Predict #############################
        # Predict the state estimate at time k based on the state 
        # estimate at time k-1 and the control input applied at time k-1.

        state_estimate_k = self.A_k_minus_1 @ (
                self.state_estimate_k_minus_1) + (
                self.getB(self.state_estimate_k_minus_1[2],self.dk)) @ (
                control_vector_k_minus_1) + (
                self.process_noise_v_k_minus_1)
                
        print(f'State Estimate Before EKF={state_estimate_k}')
                
        # Predict the state covariance estimate based on the previous
        # covariance and some noise
        P_k = self.A_k_minus_1 @ self.P_k_minus_1 @ self.A_k_minus_1.T + (self.Q_k)
            
        ################### Update (Correct) ##########################
        # Calculate the difference between the actual sensor measurements
        # at time k minus what the measurement model predicted 
        # the sensor measurements would be for the current timestep k.
        measurement_residual_y_k = z_k_observation_vector - (
                (self.H_k @ state_estimate_k) + (self.sensor_noise_w_k))
    
        print(f'Observation={z_k_observation_vector}')
                
        # Calculate the measurement residual covariance
        # see if camera gives good result to see which R matrix to use
        if(np.array_equal(z_k_observation_vector, np.zeros(3))):
            S_k = self.H_k @ P_k @ self.H_k.T + self.R_no_cam
        else:
            S_k = self.H_k @ P_k @ self.H_k.T + self.R_k
            
        # Calculate the near-optimal Kalman gain
        # We use pseudoinverse since some of the matrices might be
        # non-square or singular.
        K_k = P_k @ self.H_k.T @ np.linalg.pinv(S_k)
            
        # Calculate an updated state estimate for time k
        state_estimate_k =  np.round(state_estimate_k + (K_k @ measurement_residual_y_k)).astype(int)
        
        # Update the state covariance estimate for time k
        P_k = P_k - (K_k @ self.H_k @ P_k)
        
        # Print the best (near-optimal) estimate of the current state of the robot
        print(f'State Estimate After EKF={state_estimate_k}')
    
        # Return the updated state and covariance estimates
        return state_estimate_k, P_k

    def filter(self, camera_pos, speed, dt):
        """
        Runs one step of the EKF. The only method to use in this class. 
        Args:
            camera_pos: np.array - position given by the camera. [pos_x, pos_y, orientation] in pixels, pixels, degrees
            speed: np.array - speed of the robot when filter is called. [linear_speed, angular speed]
                                pixels/sec and degrees/sec (+:trigonometric/-:clockwise)
            dt: time interval since last time filter was called
        """                    
        self.dk = dt
        # Run the Extended Kalman Filter and store the 
        # near-optimal state and covariance estimates
        optimal_state_estimate_k, covariance_estimate_k = self.ekf(
            camera_pos, # Most recent sensor measurement
            speed) # Our most recent control input)
        
        # Get ready for the next timestep by updating the variable values
        self.state_estimate_k_minus_1 = optimal_state_estimate_k
        self.P_k_minus_1 = covariance_estimate_k

        return optimal_state_estimate_k[0], optimal_state_estimate_k[1], optimal_state_estimate_k[2]

    def get_pos(self):
        return self.state_estimate_k_minus_1[0], self.state_estimate_k_minus_1[1], self.state_estimate_k_minus_1[2]