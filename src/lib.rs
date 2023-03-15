use ode_solvers::dopri5::*;
use ode_solvers::*;

type State = Vector6<f64>;
type Time = f64;

struct Orbit {
    mu: f64,
}

impl System<State> for Orbit {
    /**
    Kepler Orbit Equations of motion.

    # Arguments
       * '_t' - The moment in time corresponding to a specific state.
       * 'y' - The state vector
       * 'dy' -  The change in the state vector
    */
    fn system(&self, _t: Time, y: &State, dy: &mut State) {
        let denominator: f64 = (y[0].powf(2.0) + y[1].powf(2.0) + y[2].powf(2.0)).powf(3.0 / 2.0);
        dy[0] = y[3];
        dy[1] = y[4];
        dy[2] = y[5];
        dy[3] = -self.mu * y[0] / denominator;
        dy[4] = -self.mu * y[1] / denominator;
        dy[5] = -self.mu * y[2] / denominator;
    }
}

/**
Propagate a state vector for a given time of flight.

# Arguments
* `state_vector` - The 1x6 array of cartesian position and velocity elements.

# Returns
* `final_position` - The final position of the spacecraft after propagation.
 */
fn propagate(state_vector: Vector6<f64>) -> Vec<Vector6<f64>> {
    let system = Orbit { mu: 1.327e11 }; // mu km-3/s-2

    let rtol: f64 = 1e-6;
    let atol: f64 = 1e-8;

    let time_start = 0.0;
    let time_of_flight: f64 = 6.189400 * 86400.0;

    let mut stepper = Dopri5::new(
        system,
        time_start,
        time_of_flight,
        10.0,
        state_vector,
        rtol,
        atol,
    );
    stepper
        .integrate()
        .expect("ERROR: Unable to integrate provided parameters.");

    let y_out = stepper.y_out();
    y_out.to_vec()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_relatively_eq(num_one: &f64, num_two: &f64, epsilon: f64) {
        let diff = (num_two - num_one).abs();
        assert!(diff <= epsilon);
    }

    #[test]
    fn should_integrate() {
        let y0 = State::new(
            -131386230.977293,
            69971484.9501445,
            -718889.822774674,
            -1.745306e+01,
            -2.843202e+01,
            -6.151334e-01,
        );
        let result = propagate(y0);
        let final_value = result.last().unwrap();

        // Position Vectors
        assert_relatively_eq(&1.0, &1.1, 0.2);
        assert_eq!(final_value[0], -139952726.88639712);
        assert_eq!(final_value[1], 54397052.70895448);
        assert_eq!(final_value[2], -1043117.4394616715);

        // Velocity Vectors
        assert_eq!(final_value[3], -14.567781976116068);
        assert_eq!(final_value[4], -29.755390699878056);
        assert_eq!(final_value[5], -0.5964095238080424);
    }
}
