using CommunityToolkit.HighPerformance;

namespace TensorWeaver.InputProcessing;

public interface Resizer
{
	void Resize<TPixel>(ReadOnlySpan2D<TPixel> source, Span2D<TPixel> target);
}