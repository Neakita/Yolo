namespace TensorWeaver.OutputProcessing;

public interface BoundedOutputProcessor<T> : OutputProcessor<T>
{
	float MaximumIoU { get; set; }
}