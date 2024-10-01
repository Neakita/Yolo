namespace Yolo.OutputProcessing;

public interface BoundedOutputProcessor<out T> : OutputProcessor<T>
{
	float MaximumIoU { get; set; }
}