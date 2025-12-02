using CommunityToolkit.HighPerformance;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace TensorWeaver.InputProcessing;

public interface InputProcessor<TPixel>
{
	void ProcessInput(ReadOnlySpan2D<TPixel> pixels, DenseTensor<float> tensor);
}