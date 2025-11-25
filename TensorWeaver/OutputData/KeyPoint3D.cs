namespace TensorWeaver.OutputData;

public struct KeyPoint3D
{
	public Vector2D<float> Position { get; }
	public float Confidence { get; }

	public KeyPoint3D(Vector2D<float> position, float confidence)
	{
		Position = position;
		Confidence = confidence;
	}
}