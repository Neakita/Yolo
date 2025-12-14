using TensorWeaver.InputProcessing;

namespace TensorWeaver;

public static class InputProcessorExtensions
{
	public static ResizingInputProcessor<TPixel> WithResizing<TPixel>(this InputProcessor<TPixel> inputProcessor, Resizer resizer)
	{
		return new ResizingInputProcessor<TPixel>(inputProcessor, resizer);
	}

	public static ResizingInputProcessor<TPixel> WithResizing<TPixel>(this InputProcessor<TPixel> inputProcessor)
	{
		return inputProcessor.WithResizing(new NearestNeighbourImageResizer());
	}
}