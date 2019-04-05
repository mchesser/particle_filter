extern crate rand;
extern crate rayon;

use std::mem;

use rand::Rng;
use rayon::prelude::*;

/// A structure for efficiently reusing particle buffers
struct ParticleBuffer<P> {
    current_particles: Vec<P>,
    previous_particles: Vec<P>,
}

impl<P> ParticleBuffer<P> {
    fn step(&mut self) {
        mem::swap(&mut self.current_particles, &mut self.previous_particles);
        self.current_particles.clear();
    }
}

/// A basic particle implementation generic over the particle and measurement type
pub struct ParticleFilter<PARTICLE, MEASUREMENT: ?Sized, F1, F2, F3>
where
    F3: for<'a> FnMut(PARTICLE, &'a MEASUREMENT) -> f32,
{
    particles: ParticleBuffer<PARTICLE>,
    weights: Vec<f32>,
    propagation_function: F1,
    noise_function: F2,
    weight_function: F3,

    _measurement: std::marker::PhantomData<*const MEASUREMENT>,
}

impl<PARTICLE, MEASUREMENT, F1, F2, F3> ParticleFilter<PARTICLE, MEASUREMENT, F1, F2, F3>
where
    PARTICLE: Copy + Clone,
    MEASUREMENT: ?Sized,
    F1: FnMut(PARTICLE, f32) -> PARTICLE,
    F2: FnMut(PARTICLE, f32) -> PARTICLE,
    F3: for<'a> FnMut(PARTICLE, &'a MEASUREMENT) -> f32,
{
    /// Create a new particle filter given and initial particle distribution.
    ///
    /// The initial particle distribution is used to determine the number of particles to be used
    /// throughout the particle filter's lifetime.
    pub fn new(
        initial_particles: Vec<PARTICLE>,
        propagation_function: F1,
        noise_function: F2,
        weight_function: F3,
    ) -> ParticleFilter<PARTICLE, MEASUREMENT, F1, F2, F3> {
        let num_particles = initial_particles.len();

        ParticleFilter {
            particles: ParticleBuffer {
                current_particles: initial_particles,
                previous_particles: Vec::with_capacity(num_particles),
            },
            weights: Vec::with_capacity(num_particles),
            propagation_function,
            noise_function,
            weight_function,

            _measurement: std::marker::PhantomData,
        }
    }

    /// Perform a step in the particle filter
    pub fn step(&mut self, measurement: &MEASUREMENT, dt: f32) -> f32 {
        self.weights.clear();
        let mut sum_weights = 0.0;

        for particle in &mut self.particles.current_particles {
            // Update the particle
            *particle = (self.propagation_function)(*particle, dt);

            // Compute the particle's new weight
            let weight = (self.weight_function)(*particle, &measurement);
            sum_weights += weight;
            self.weights.push(weight);
        }

        // Normalise weights
        for weight in &mut self.weights {
            *weight /= sum_weights
        }

        let num_particles = self.get_particles().len();
        self.resample(num_particles);

        for particle in &mut self.particles.current_particles {
            *particle = (self.noise_function)(*particle, dt);
        }

        sum_weights / self.weights.len() as f32
    }

    /// Return a slice to the current list of particles.
    pub fn get_particles(&self) -> &[PARTICLE] {
        &self.particles.current_particles
    }

    /// Perform a resampling step using the low-variance resampling method.
    fn resample(&mut self, num_particles: usize) {
        self.particles.step();

        let mut rng = rand::thread_rng();

        let step_factor = 1.0 / num_particles as f64;
        let start = rng.gen::<f64>() * step_factor;

        let mut i = 0;
        let mut cumulative_weight = self.weights[i] as f64;

        // Generate a new particle for each old particle
        for particle_count in 0..num_particles {
            let current_weight = start + particle_count as f64 * step_factor;

            while i + 1 < num_particles && cumulative_weight < current_weight {
                i += 1;
                cumulative_weight += self.weights[i] as f64;
            }

            self.particles.current_particles.push(self.particles.previous_particles[i]);
        }
    }

    /// Merges the particles from one particle filter with the current particle filter
    pub fn merge_particles(&mut self, other: &[PARTICLE], ratio: f32) {
        assert!(ratio >= 0.0 && ratio <= 1.0);

        let num_particles = self.particles.current_particles.len();

        let new_particles = ((ratio * num_particles as f32) as usize).min(other.len());
        let base_particles = num_particles - new_particles;

        self.particles.current_particles.truncate(base_particles);
        self.particles.current_particles.extend_from_slice(&other[..new_particles]);
    }
}

/// A utility trait for managing the filter as a trait object
pub trait Filter {
    type Particle;
    type Measurement: ?Sized;

    /// Get a view of the current particles
    fn get_particles(&self) -> &[Self::Particle];

    /// Perform a single step of the particle filter
    fn step(&mut self, measurement: &Self::Measurement, dt: f32) -> f32;

    /// Merges one set of particles with another (used for reinitialization)
    fn merge_particles(&mut self, other: &[Self::Particle], ratio: f32);
}

impl<F1, F2, F3, S, M> Filter for ParticleFilter<S, M, F1, F2, F3>
where
    S: Copy + Clone,
    M: ?Sized,
    F1: FnMut(S, f32) -> S,
    F2: FnMut(S, f32) -> S,
    F3: for<'a> FnMut(S, &'a M) -> f32,
{
    type Particle = S;
    type Measurement = M;

    fn get_particles(&self) -> &[S] {
        self.get_particles()
    }

    fn step(&mut self, measurement: &M, dt: f32) -> f32 {
        self.step(measurement, dt)
    }

    fn merge_particles(&mut self, other: &[S], ratio: f32) {
        self.merge_particles(other, ratio);
    }
}

impl<PARTICLE, MEASUREMENT, F1, F2, F3> ParticleFilter<PARTICLE, MEASUREMENT, F1, F2, F3>
where
    PARTICLE: Copy + Clone + Send + Sync,
    MEASUREMENT: ?Sized + Sync,
    F1: Fn(PARTICLE, f32) -> PARTICLE + Send + Sync,
    F2: Fn(PARTICLE, f32) -> PARTICLE + Send + Sync,
    F3: for<'a> Fn(PARTICLE, &'a MEASUREMENT) -> f32 + Send + Sync,
{
    /// Perform a step in the particle filter running calculations in parallel.
    pub fn parallel_step(&mut self, measurement: &MEASUREMENT, dt: f32) -> f32 {
        let weight = self.parallel_step_inner(measurement, dt);

        let num_particles = self.get_particles().len();
        self.resample(num_particles);

        // Apply noise function to particles
        let noise_function = &mut self.noise_function;
        let current_particles = &mut self.particles.current_particles;
        current_particles.par_iter_mut().for_each(|p| {
            *p = (noise_function)(*p, dt);
        });

        weight
    }

    /// Inner function for parallel step.
    fn parallel_step_inner(&mut self, measurement: &MEASUREMENT, dt: f32) -> f32 {
        let propagation_function = &mut self.propagation_function;

        // Update particles
        let current_particles = &mut self.particles.current_particles;
        current_particles.par_iter_mut().for_each(|p| {
            *p = (propagation_function)(*p, dt);
        });

        // Calculate new weights
        let weight_function = &mut self.weight_function;
        let mut weights = &mut self.weights;
        current_particles
            .par_iter()
            .map(|p| (weight_function)(*p, measurement))
            .collect_into_vec(&mut weights);

        // Normalise weights
        let num_weights = weights.len();
        let weight_sum = weights.iter().fold(0.0, |acc, x| acc + x);
        for weight in weights {
            *weight /= weight_sum
        }

        weight_sum / num_weights as f32
    }
}

/// A utility trait for managing the filter as a trait object
pub trait ParallelFilter {
    type Particle;
    type Measurement: ?Sized;

    /// Get a view of the current particles
    fn get_particles(&self) -> &[Self::Particle];

    /// Perform a single step of the particle filter
    fn step(&mut self, measurement: &Self::Measurement, dt: f32) -> f32;
}

impl<F1, F2, F3, S, M> ParallelFilter for ParticleFilter<S, M, F1, F2, F3>
where
    S: Copy + Clone + Send + Sync,
    M: ?Sized + Sync,
    F1: Fn(S, f32) -> S + Send + Sync,
    F2: Fn(S, f32) -> S + Send + Sync,
    F3: for<'a> Fn(S, &'a M) -> f32 + Send + Sync,
{
    type Particle = S;
    type Measurement = M;

    fn get_particles(&self) -> &[S] {
        self.get_particles()
    }

    fn step(&mut self, measurement: &M, dt: f32) -> f32 {
        self.parallel_step(measurement, dt)
    }
}
