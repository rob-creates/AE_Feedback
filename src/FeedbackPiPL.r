#include "AEConfig.h"
#include "AE_EffectVers.h"
#include "AE_General.r"

resource 'PiPL' (16000) {
	{
		Kind { AEEffect },
		Name { "Feedback" },
		Category { "Stylize" },
		CodeMacARM64 { "EffectMain" },
		AE_PiPL_Version { 2, 0 },
		AE_Effect_Spec_Version { PF_PLUG_IN_VERSION, PF_PLUG_IN_SUBVERS },
		AE_Effect_Version { 524288 /* 1.0 */ },
		AE_Effect_Info_Flags { 0 },
		/* Must match GlobalSetup in Feedback.mm:
		   WIDE_TIME_INPUT (1<<1) | I_EXPAND_BUFFER (1<<9) | DEEP_COLOR_AWARE (1<<25)
		   | SEND_UPDATE_PARAMS_UI (1<<26) */
		AE_Effect_Global_OutFlags { 0x6000202 },
		/* SMART_RENDER (1<<10) | FLOAT_COLOR_AWARE (1<<12) | AUTOMATIC_WIDE_TIME_INPUT (1<<17)
		   | I_MIX_GUID_DEPENDENCIES (1<<21) | GPU_RENDER_F32 (1<<25) | THREADED_RENDERING (1<<27) */
		AE_Effect_Global_OutFlags_2 { 0xA221400 },
		AE_Effect_Match_Name { "RPAY Feedback" },
		AE_Reserved_Info { 0 },
		AE_Effect_Support_URL { "https://robpayne.co.uk" }
	}
};
