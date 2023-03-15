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
fn propagate(state_vector: Vector6<f64>, time_of_flight_sec: f64) -> Vec<Vector6<f64>> {
    let system = Orbit { mu: 1.327e11 }; // mu km-3/s-2

    let rtol: f64 = 1e-6;
    let atol: f64 = 1e-8;

    let time_start = 0.0;
    let mut stepper = Dopri5::new(
        system,
        time_start,
        time_of_flight_sec,
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
    use parameterized::parameterized;

    #[parameterized(input_state = {
        State::new(
        -1.236327193104345E+08,
        -1.683146780978357E+08,
        -1.716864100448400E+07,
        1.732704689147723E+01,
        -1.411175177586654E+01,
        -1.496047022540958E+00,
        ),
        State::new(
        -1.236327193104345E+08,
        -1.683146780978357E+08,
        -1.716864100448400E+07,
        1.732704689147723E+01,
        -1.411175177586654E+01,
        -1.496047022540958E+00,
        )
    }, expected_out_state = {
        State::new(
        -1.221290282017822E+08,
        -1.695247481033071E+08,
        -1.729696027041973E+07,
        1.748026039062854E+01,
        -1.389893159180101E+01,
        -1.474285282341534E+00
        ),
        State::new(
        -1.206121920408666E+08,
        -1.707163797188062E+08,
        -1.742339387827656E+07,
        1.763133062266370E+01,
        -1.368493924400783E+01,
        -1.452397742856931E+00,
        )},
    time_of_flight = { 1440.0 * 60.0, 2.0 * 1440.0 * 60.0},
    eps_pos = {200., 500.0},
    eps_vel = {0.5, 0.5}
    )]
    fn should_integrate(
        input_state: State,
        expected_out_state: State,
        time_of_flight: f64,
        eps_pos: f64,
        eps_vel: f64,
    ) {
        let result = propagate(input_state, time_of_flight);
        let final_value = result.last().unwrap();

        // Position Vectors
        assert_relatively_eq(final_value[0], expected_out_state[0], eps_pos);
        assert_relatively_eq(final_value[1], expected_out_state[1], eps_pos);
        assert_relatively_eq(final_value[2], expected_out_state[2], eps_pos);

        // Velocity Vectors
        assert_relatively_eq(final_value[3], expected_out_state[3], eps_vel);
        assert_relatively_eq(final_value[4], expected_out_state[4], eps_vel);
        assert_relatively_eq(final_value[5], expected_out_state[5], eps_vel);
    }

    fn assert_relatively_eq(num_one: f64, num_two: f64, epsilon: f64) {
        let diff = (num_two - num_one).abs();
        assert!(diff <= epsilon);
    }
}
