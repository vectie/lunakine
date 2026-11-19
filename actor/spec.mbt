///|
pub suberror ActorError {
  InvalidActor(String)
} derive(Debug)

///|
pub extend ActorError with @debug.Debug::{to_repr}

///|
pub(all) struct Normalization {
  mean : Array[Double]
  variance : Array[Double]
  epsilon : Double
} derive(ToJson, FromJson, Debug)

///|
pub extend Normalization with ToJson::{to_json}

///|
pub extend Normalization with @json.FromJson::{from_json}

///|
pub extend Normalization with @debug.Debug::{to_repr}

///|
/// One batch-one ELU encoder/decoder program. Width arrays include input and
/// output widths. The encoder output is tanh-activated; decoder output is linear.
pub(all) struct Spec {
  format : String
  history_frames : Int32
  observation_width : Int32
  encoder : Array[Int32]
  decoder : Array[Int32]
  observation_norm : Normalization
  goal_norm : Normalization
  action_mask : Array[Double]
} derive(ToJson, FromJson, Debug)

///|
pub extend Spec with ToJson::{to_json}

///|
pub extend Spec with @json.FromJson::{from_json}

///|
pub extend Spec with @debug.Debug::{to_repr}

///|
pub fn finite(value : Double) -> Bool {
  value == value && value - value == 0.0
}

///|
fn Normalization::validate(
  self : Normalization,
  width : Int32,
) -> Unit raise ActorError {
  guard self.mean.length() == width &&
    self.variance.length() == width &&
    self.mean.all(finite) &&
    self.variance.all(x => finite(x) && x >= 0.0) &&
    finite(self.epsilon) &&
    self.epsilon > 0.0 else {
    raise InvalidActor("invalid normalization")
  }
}

///|
pub fn Spec::parameter_count(self : Spec) -> Int32 raise ActorError {
  let mut count = 0
  for widths in [self.encoder, self.decoder] {
    guard widths.length() >= 2 && widths.all(x => x > 0) else {
      raise InvalidActor("network requires positive input/output widths")
    }
    for i in 0..<(widths.length() - 1) {
      let input = widths[i]
      let output = widths[i + 1]
      guard input < 536870911 && output <= (536870911 - count) / (input + 1) else {
        raise InvalidActor("parameter byte count exceeds native buffer range")
      }
      count += (input + 1) * output
    }
  }
  count
}

///|
pub fn Spec::validate(self : Spec) -> Unit raise ActorError {
  guard self.format == "lunakine.elu-actor.v1" &&
    self.history_frames > 0 &&
    self.observation_width > 0 &&
    self.history_frames <= 2147483647 / self.observation_width else {
    raise InvalidActor("invalid actor format or observation dimensions")
  }
  ignore(self.parameter_count())
  let latent = self.encoder[self.encoder.length() - 1]
  let history = self.history_frames * self.observation_width
  guard latent <= 2147483647 - history &&
    self.decoder[0] == latent + history &&
    self.action_mask.length() == self.decoder[self.decoder.length() - 1] &&
    self.action_mask.all(x => x == 0.0 || x == 1.0) else {
    raise InvalidActor("decoder input or action mask mismatch")
  }
  self.observation_norm.validate(self.observation_width)
  self.goal_norm.validate(self.encoder[0])
}

///|
pub fn Spec::history_width(self : Spec) -> Int32 {
  self.history_frames * self.observation_width
}

///|
pub fn Spec::goal_width(self : Spec) -> Int32 {
  self.encoder[0]
}

///|
pub fn Spec::action_width(self : Spec) -> Int32 {
  self.action_mask.length()
}
