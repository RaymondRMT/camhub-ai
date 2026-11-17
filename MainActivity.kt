package com.camhub.installer

import android.app.Activity
import android.content.Intent
import android.content.pm.PackageManager
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.widget.Button
import android.widget.TextView
import android.widget.Toast

class MainActivity : Activity() {

    private val INSTALL_SCRIPT_URL = "https://raw.githubusercontent.com/RaymondRMT/camhub-ai/main/setup_camhub.sh"

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        val btnInstall = findViewById<Button>(R.id.btnInstall)
        val btnOpenWeb = findViewById<Button>(R.id.btnOpenWeb)
        val tvStatus = findViewById<TextView>(R.id.tvStatus)

        btnInstall.setOnClickListener {
            if (!isTermuxInstalled()) {
                tvStatus.text = "Chưa phát hiện Termux! Đang mở trang tải Termux (F-Droid)..."
                startActivity(Intent(Intent.ACTION_VIEW, Uri.parse("https://f-droid.org/packages/com.termux/")))
                return@setOnClickListener
            }

            tvStatus.text = "Đang kích hoạt Termux và tải các gói dữ liệu hệ thống...\nVui lòng giữ màn hình sáng trong khoảng 2-3 phút."
            triggerTermuxInstallation()
        }

        btnOpenWeb.setOnClickListener {
            startActivity(Intent(Intent.ACTION_VIEW, Uri.parse("http://localhost:8000")))
        }
    }

    private fun isTermuxInstalled(): Boolean {
        return try {
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
                packageManager.getPackageInfo("com.termux", PackageManager.PackageInfoFlags.of(0))
            } else {
                @Suppress("DEPRECATION")
                packageManager.getPackageInfo("com.termux", 0)
            }
            true
        } catch (e: PackageManager.NameNotFoundException) {
            false
        }
    }

    private fun triggerTermuxInstallation() {
        try {
            val launchIntent = packageManager.getLaunchIntentForPackage("com.termux")
            if (launchIntent != null) startActivity(launchIntent)

            Handler(Looper.getMainLooper()).postDelayed({
                try {
                    val cmdIntent = Intent("com.termux.RUN_COMMAND").apply {
                        setClassName("com.termux", "com.termux.app.RunCommandService")
                        putExtra("com.termux.RUN_COMMAND_PATH", "/data/data/com.termux/files/usr/bin/bash")
                        putExtra("com.termux.RUN_COMMAND_ARGUMENTS", arrayOf("-c", "curl -fsSL $INSTALL_SCRIPT_URL | bash"))
                        putExtra("com.termux.RUN_COMMAND_WORKDIR", "/data/data/com.termux/files/home")
                        putExtra("com.termux.RUN_COMMAND_BACKGROUND", false)
                    }
                    if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
                        startForegroundService(cmdIntent)
                    } else {
                        startService(cmdIntent)
                    }
                } catch (e: Exception) {
                    Toast.makeText(
                        this,
                        "Hãy bật 'allow-external-apps=true' trong ~/.termux/termux.properties và cấp quyền RUN_COMMAND!",
                        Toast.LENGTH_LONG
                    ).show()
                }
            }, 1500)
        } catch (e: Exception) {
            Toast.makeText(this, "Không thể kích hoạt Termux: ${e.message}", Toast.LENGTH_LONG).show()
        }
    }
}
