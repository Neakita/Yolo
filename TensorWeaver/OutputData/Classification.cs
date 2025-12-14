namespace TensorWeaver.OutputData;

public readonly struct Classification
{
	public byte ClassId { get; }
	public float Confidence { get; }

	public Classification(byte classId, float confidence)
	{
		ClassId = classId;
		Confidence = confidence;
	}
}