namespace Yolo;

public sealed class Classification : Prediction
{
	public Classification(ushort classId, float confidence) : base(classId, confidence)
	{
	}
}