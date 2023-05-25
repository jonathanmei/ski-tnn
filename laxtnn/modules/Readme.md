These modules are the main components of SKI-TNN and FD-TNN.

![tnn](https://github.com/jonathanmei/ski-tnn-private/assets/2014727/2150406d-a1ee-49e7-855f-76495af5cf13)

The core components are
1. The GTU, which does channel mixing via the linear operators and token mixing via the TNO. This is in gtu.py
2. The TNO, which does token mixing. The original TNO from the TNN paper is in tno.py. Our SKI-TNO from our paper is in skitno_inv_time.py and is class SKITnoInvTime. Our FD-TNO is in tno_fd.py as class TnoFD.
  a. SKI-TNO uses a 1d conv for the sparse component and SKI for the low rank component. We have a class for handling such components, Sltno, which SKITnoInvTime inherits from.
4. Both the original TNO and our FD-TNO use an RPE MLP, which maps scalars (representing times or frequencies) to vectors, one for each embedding dimension. The code for this is in rpe.py
