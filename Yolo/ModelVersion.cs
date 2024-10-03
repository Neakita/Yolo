namespace Yolo;

public readonly struct ModelVersion
{
	public byte Major { get; }

	public ModelVersion(byte major)
	{
		Major = major;
	}
}