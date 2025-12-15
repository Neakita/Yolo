namespace TensorWeaver.OutputData;

public readonly struct Classification
{
	public ushort ClassId { get; }
	public float Confidence { get; }

	public Classification(ushort classId, float confidence)
	{
		ClassId = classId;
		Confidence = confidence;
	}
}