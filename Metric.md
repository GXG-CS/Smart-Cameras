# Abstracting Smart Camera Hardware Capabilities Using ML

## Hardware Metrics in Mathematical Notation:

- S<sub>read</sub>: Storage read speed (MB/s)
- S<sub>cap</sub>: Storage capacity (GB)
- R<sub>cap</sub>: RAM capacity (GB)
- R<sub>speed</sub>: RAM speed (MHz or GHz)
- P<sub>speed</sub>: CPU processing speed (GHz)
- P<sub>cores</sub>: Number of CPU cores
- P<sub>consumption</sub>: Power consumption (W)
- T<sub>manage</sub>: Thermal management capacity (°C)

## Machine Learning Process with Attached Metrics:

### Data Loading and Preprocessing:
- Load: S<sub>read</sub>
- Preprocessing memory: R<sub>cap</sub>, R<sub>speed</sub>

### Model Selection and Initialization:
- Model storage: S<sub>cap</sub>

### Model Training:
- Computation: P<sub>speed</sub>, P<sub>cores</sub>
- Batch processing: R<sub>cap</sub>, R<sub>speed</sub>
- Model checkpoint storage: S<sub>read</sub>, S<sub>cap</sub>

### Model Evaluation:
- Computation: P<sub>speed</sub>, P<sub>cores</sub>
- Evaluation dataset in memory: R<sub>cap</sub>, R<sub>speed</sub>

### Model Storage & Versioning:
- Model save/load: S<sub>read</sub>, S<sub>cap</sub>

### Deployment & Inference:
- Inference computation: P<sub>speed</sub>, P<sub>cores</sub>
- Model in memory: R<sub>cap</sub>, R<sub>speed</sub>

### Feedback and Iterative Improvement:
- Feedback storage: S<sub>cap</sub>

### Optimizations:
- Computation: P<sub>speed</sub>, P<sub>cores</sub>
- Memory during optimization: R<sub>cap</sub>, R<sub>speed</sub>

### Throughout all the phases:
- Power usage: P<sub>consumption</sub>
- Thermal limits: T<sub>manage</sub>

## Mathematical Relationship:

The ML process on a smart camera, given the absence of a GPU, can be represented as:

ML<sub>process</sub> = f(S<sub>read</sub>, S<sub>cap</sub>, R<sub>cap</sub>, R<sub>speed</sub>, P<sub>speed</sub>, P<sub>cores</sub>, P<sub>consumption</sub>, T<sub>manage</sub>)


