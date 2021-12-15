Shader "Unlit/Display"
{
    Properties
    {
        _MainTex ("Texture", 2D) = "white" {}
    }
    SubShader
    {
        Tags { "RenderType"="Opaque" }
        LOD 100

        Pass
        {
            CGPROGRAM
            #pragma vertex vert
            #pragma fragment frag
            // make fog work
            #pragma multi_compile_fog

            #include "UnityCG.cginc"

            struct appdata
            {
                float4 vertex : POSITION;
                float2 uv : TEXCOORD0;
            };

            struct v2f
            {
                float2 uv : TEXCOORD0;
                float4 color : COLOR;
                float4 vertex : SV_POSITION;
            };

            sampler2D _MainTex;
            float4 _MainTex_ST;
            uniform StructuredBuffer<float> _vertices;
            uniform StructuredBuffer<float3> _normal;
            uniform StructuredBuffer<int> _idx;
            float4  _LightColor0;

            v2f vert (appdata v,uint vid: SV_VertexID)
            {
                v2f o;
                //o.vertex = UnityObjectToClipPos(v.vertex);
                o.uv = TRANSFORM_TEX(v.uv, _MainTex);
                o.vertex = UnityObjectToClipPos(float4(_vertices[_idx[vid]*3+0], _vertices[_idx[vid] * 3 + 1], _vertices[_idx[vid] * 3 + 2],1));
                int nid = vid / 3;
                float4 vertex_normal = float4(_normal[nid], 1);
                float3 NormalDirection = normalize(vertex_normal.xyz);
                float4 AmbientLight = UNITY_LIGHTMODEL_AMBIENT;
                float4 LightDirection = normalize(_WorldSpaceLightPos0);
                o.color = saturate(dot(LightDirection, NormalDirection)) * _LightColor0 + AmbientLight;
                return o;
            }

            fixed4 frag(v2f i) : SV_Target
            {
                return tex2D(_MainTex, i.uv) *i.color;
            }
            ENDCG
        }
    }
}
