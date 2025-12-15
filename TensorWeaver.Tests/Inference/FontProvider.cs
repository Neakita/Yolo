using SixLabors.Fonts;

namespace TensorWeaver.Tests.Inference;

internal static class FontProvider
{
	public static Font Font { get; }

	static FontProvider()
	{
		foreach (var possibleFontFamilyName in PossibleFontFamilyNames)
		{
			if (!SystemFonts.TryGet(possibleFontFamilyName, out var fontFamily))
				continue;
			Font = fontFamily.CreateFont(12, FontStyle.Bold);
			return;
		}
		throw new InvalidOperationException("Can't find system font");
	}

	private static readonly IEnumerable<string> PossibleFontFamilyNames = ["FreeMono", "Noto Sans", "Arial"];
}